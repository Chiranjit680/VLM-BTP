"""Agent logic: a tool-using loop shared by the global and tile agents, plus the aggregator."""
import json
import re
import time

from PIL import Image

from . import config, logs, store
from .model import get_model
from .mcp_client import get_tool_client

TOOL_SYSTEM = """You are a careful biomedical image analyst.
You may transform the image you are looking at with these tools:
{tools}

Reply with EXACTLY one JSON object and nothing else:
  {{"tool": "<name>", "args": {{...}}}}   to apply a tool (you then see the result), or
  {{"final": "<your findings>"}}          when you are done.
Only use a tool if it would help; be factual and do not invent findings."""


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_tool_loop(image: Image.Image, task: str, agent: str = "agent"):
    """ReAct-style loop.

    Returns (final_text, steps). `steps` is this agent's reasoning: one entry per
    model call, holding the raw reply, its latency, and any tool call it made
    with the path of the image that tool produced.
    """
    log = logs.get(agent)
    vlm, tools = get_model(), get_tool_client()
    run = store.current()
    system = TOOL_SYSTEM.format(tools=tools.describe_tools())
    current, steps, history = image, [], ""
    log.info("start: image %dx%d, up to %d tool steps", *image.size, config.MAX_TOOL_STEPS)
    log.debug("task: %s", logs.short(task, 400))

    def done(report):
        log.info("done after %d tool call(s): %s",
                 sum("tool" in st for st in steps), logs.short(report))
        run.record_agent(agent, report, steps)
        return report, steps

    for step in range(config.MAX_TOOL_STEPS + 1):
        force_final = step == config.MAX_TOOL_STEPS
        prompt = f"Task: {task}\n{history}"
        if force_final:
            prompt += '\nNo more tool calls allowed. Reply with {"final": "..."}.'

        t0 = time.perf_counter()
        reply = vlm.chat(prompt, current, system=system)
        seconds = time.perf_counter() - t0
        record = {"step": step + 1, "seconds": round(seconds, 2), "reply": reply}
        steps.append(record)
        log.info("step %d/%d: model replied in %.1fs%s", step + 1, config.MAX_TOOL_STEPS + 1,
                 seconds, " (final forced)" if force_final else "")
        log.debug("step %d raw reply: %s", step + 1, logs.short(reply, 500))
        action = _parse_json(reply)

        if action is None:  # model ignored the format; treat raw text as the answer
            log.warning("step %d: reply was not JSON, using it verbatim as the answer", step + 1)
            record["final"] = reply
            return done(reply)
        if "final" in action:
            record["final"] = str(action["final"])
            return done(record["final"])
        if force_final:
            log.warning("step %d: asked for a tool after the limit, keeping the raw reply", step + 1)
            record["final"] = reply
            return done(reply)

        name, args = action.get("tool"), action.get("args", {})
        record.update(tool=name, args=args)
        log.info("step %d: calling tool %s(%s)", step + 1, name, args)
        try:
            t0 = time.perf_counter()
            current = tools.call(name, current, args)
            log.info("step %d: %s -> image %dx%d in %.2fs",
                     step + 1, name, *current.size, time.perf_counter() - t0)
            record["result_image"] = run.save_image(f"{agent} step{step + 1} {name}", current)
            history += f"\n[step {step + 1}] applied {name}({args}); the image shown is now the result."
        except ValueError as e:
            log.warning("step %d: tool %s failed: %s", step + 1, name, logs.short(e, 200))
            record["error"] = str(e)
            history += f"\n[step {step + 1}] tool call failed: {e}"
    log.error("loop ended without an answer after %d tool call(s)", len(steps))
    return done("")


def global_agent(image: Image.Image, question: str):
    task = (
        "You see the WHOLE image (downscaled). Describe the modality, anatomy, overall layout "
        "and any salient abnormalities, then say what matters for the question.\n"
        f"Question: {question}"
    )
    return run_tool_loop(image, task, agent="global")


def tile_agent(tile: Image.Image, tile_id: str, question: str, global_summary: str):
    task = (
        f"You see ONE TILE ({tile_id}) of a larger image at higher resolution. "
        "Report only what is visible in this tile that is relevant to the question, "
        "and say 'nothing relevant' if that is the case.\n"
        f"Global context: {global_summary}\nQuestion: {question}"
    )
    return run_tool_loop(tile, task, agent=f"tile {tile_id}")


def aggregator(question: str, global_summary: str, tile_reports: list[dict]):
    """Text-only synthesis of the global view and all tile findings."""
    reports = "\n".join(f"- {r['tile_id']}: {r['report']}" for r in sorted(tile_reports, key=lambda r: r["tile_id"]))
    prompt = (
        "You are the final reasoner. Combine the global analysis and the per-tile findings into a "
        "single coherent answer. Resolve conflicts, ignore 'nothing relevant' tiles, and mention "
        "where in the image (tile id) key evidence was seen.\n\n"
        f"Question: {question}\n\nGlobal analysis: {global_summary}\n\nTile findings:\n{reports}\n\nFinal answer:"
    )
    log = logs.get("aggregator")
    usable = [r for r in tile_reports if "nothing relevant" not in r["report"].lower()]
    log.info("merging global summary + %d tile reports (%d with findings)", len(tile_reports), len(usable))
    with logs.timed(log, "synthesising final answer"):
        answer = get_model().chat(prompt, max_new_tokens=384)
    log.info("answer: %s", logs.short(answer))
    store.current().record_agent("aggregator", answer, [{"step": 1, "seconds": 0.0, "reply": answer}])
    return answer
