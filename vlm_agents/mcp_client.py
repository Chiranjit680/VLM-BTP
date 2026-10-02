"""Synchronous facade over the MCP tool server.

The LangGraph nodes are synchronous, so the async MCP session lives on a background
event-loop thread and calls are bridged with run_coroutine_threadsafe (thread-safe).
"""
import asyncio
import atexit
import base64
import io
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from . import config, logs

log = logs.get("mcp")

_IMAGE_ARG = "image_b64"


class ToolClient:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._stop = None
        self._session = None
        self._tools = {}
        self._submit(self._start())
        atexit.register(self.close)

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _start(self):
        ready = asyncio.Event()
        self._stop = asyncio.Event()
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "vlm_agents.mcp_server"], cwd=config.ROOT
        )

        async def hold():  # keep the stdio/session context managers in one task
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    self._session = session
                    self._tools = {t.name: t for t in (await session.list_tools()).tools}
                    ready.set()
                    await self._stop.wait()

        self._task = asyncio.ensure_future(hold())
        await ready.wait()
        log.info("tool server up with %d tools: %s", len(self._tools), ", ".join(self._tools))

    def describe_tools(self) -> str:
        lines = []
        for t in self._tools.values():
            params = [p for p in t.input_schema.get("properties", {}) if p != _IMAGE_ARG]
            lines.append(f"- {t.name}({', '.join(params)}): {t.description}")
        return "\n".join(lines)

    def call(self, name: str, image: Image.Image, args: dict) -> Image.Image:
        if name not in self._tools:
            raise ValueError(f"unknown tool '{name}'")
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        payload = {**(args or {}), _IMAGE_ARG: base64.b64encode(buf.getvalue()).decode()}
        log.debug("call %s(%s) on a %dx%d image (%.0f KiB payload)",
                  name, args, *image.size, len(payload[_IMAGE_ARG]) / 1024)
        result = self._submit(self._session.call_tool(name, payload))
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise ValueError(f"{name} failed: {text[:200]}")
        for c in result.content:
            if c.type == "image":
                return Image.open(io.BytesIO(base64.b64decode(c.data)))
        raise ValueError(f"{name} returned no image")

    def close(self):
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._stop = None


_client = None


def get_tool_client() -> ToolClient:
    global _client
    if _client is None:
        _client = ToolClient()
    return _client
