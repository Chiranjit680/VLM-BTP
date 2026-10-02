"""MCP server exposing the image-transformation tools (stdio).

Stateless: each tool receives the image as base64 and returns the transformed PNG.
Run standalone with: python -m vlm_agents.mcp_server
"""
import base64
import io
import logging

from mcp.server.mcpserver import Image as MCPImage, MCPServer
from PIL import Image

from .tools.image_tools import apply_tool

# The server runs as a subprocess whose stderr lands in our terminal; its own
# INFO lines duplicate what the agent already logs, in a different format.
logging.getLogger("mcp").setLevel(logging.WARNING)

mcp = MCPServer("biomedical-image-tools")


def _run(name, image_b64, **args):
    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    out = apply_tool(name, img, args)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return MCPImage(data=buf.getvalue(), format="png")


@mcp.tool()
def zoom(image_b64: str, x0: float, y0: float, x1: float, y1: float) -> MCPImage:
    """Crop a normalised box (coords 0-1) and upscale it to inspect fine detail."""
    return _run("zoom", image_b64, x0=x0, y0=y0, x1=x1, y1=y1)


@mcp.tool()
def autocontrast(image_b64: str, cutoff: float = 1) -> MCPImage:
    """Stretch intensities; cutoff (0-20) is the percent clipped at each end."""
    return _run("autocontrast", image_b64, cutoff=cutoff)


@mcp.tool()
def equalize(image_b64: str) -> MCPImage:
    """Histogram equalisation to reveal low-contrast structures."""
    return _run("equalize", image_b64)


@mcp.tool()
def gamma(image_b64: str, value: float = 0.7) -> MCPImage:
    """Gamma correction; value <1 brightens shadows, >1 darkens (0.3-3)."""
    return _run("gamma", image_b64, value=value)


@mcp.tool()
def sharpen(image_b64: str) -> MCPImage:
    """Unsharp mask to emphasise edges and boundaries."""
    return _run("sharpen", image_b64)


@mcp.tool()
def denoise(image_b64: str) -> MCPImage:
    """Median filter to suppress speckle noise."""
    return _run("denoise", image_b64)


@mcp.tool()
def edges(image_b64: str) -> MCPImage:
    """Edge map of the image, useful for outlines and shapes."""
    return _run("edges", image_b64)


@mcp.tool()
def invert(image_b64: str) -> MCPImage:
    """Invert intensities (can make bright-on-dark findings easier to see)."""
    return _run("invert", image_b64)


if __name__ == "__main__":
    mcp.run()
