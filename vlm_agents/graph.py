"""LangGraph wiring: global agent -> 9 tile agent nodes (parallel) -> aggregator."""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image

from . import agents, config, logs, store

log = logs.get("graph")

DEFAULT_QUESTION = "Describe this image and interpret its clinically relevant findings."


class State(TypedDict, total=False):
    image: Image.Image
    question: str
    global_summary: str
    global_trace: list
    tiles: list                                        # [{"tile_id", "image"}]
    tile_reports: Annotated[list, operator.add]        # reducer merges parallel tile results
    answer: str


def make_tiles(image: Image.Image, grid=config.TILE_GRID, overlap=config.TILE_OVERLAP):
    w, h = image.size
    tw, th = w / grid, h / grid
    pad_x, pad_y = tw * overlap, th * overlap
    tiles = []
    for r in range(grid):
        for c in range(grid):
            box = (max(0, c * tw - pad_x), max(0, r * th - pad_y),
                   min(w, (c + 1) * tw + pad_x), min(h, (r + 1) * th + pad_y))
            tiles.append({"tile_id": f"r{r}c{c}", "image": image.crop(tuple(int(v) for v in box))})
    return tiles


def global_node(state: State):
    img = state["image"].convert("RGB")
    small = img.copy()
    small.thumbnail((config.GLOBAL_MAX_SIDE, config.GLOBAL_MAX_SIDE))
    log.info("node global_agent: source %dx%d, downscaled to %dx%d", *img.size, *small.size)

    run = store.current()
    run.save_image("source", img)
    run.save_image("global input", small)

    summary, trace = agents.global_agent(small, state["question"])
    tiles = make_tiles(img)
    for t in tiles:
        run.save_image(f"tile {t['tile_id']}", t["image"])
    log.info("node global_agent: cut %d tiles (%dx%d grid, %.0f%% overlap), tile size ~%dx%d",
             len(tiles), config.TILE_GRID, config.TILE_GRID, config.TILE_OVERLAP * 100,
             *tiles[0]["image"].size)
    return {"global_summary": summary, "global_trace": trace, "tiles": tiles}


def make_tile_node(tile_id: str):
    """Build the node for one tile agent; it picks its own tile out of the shared state."""
    def node(state: State):
        tile = next(t for t in state["tiles"] if t["tile_id"] == tile_id)
        log.info("node tile_agent_%s: entering", tile_id)
        report, trace = agents.tile_agent(tile["image"], tile_id, state["question"], state["global_summary"])
        log.info("node tile_agent_%s: done, %d tool call(s)", tile_id, len(trace))
        return {"tile_reports": [{"tile_id": tile_id, "report": report, "trace": trace}]}
    return node


def aggregator_node(state: State):
    log.info("node aggregator: all %d tile agents finished", len(state["tile_reports"]))
    return {"answer": agents.aggregator(state["question"], state["global_summary"], state["tile_reports"])}


def build_graph(grid=config.TILE_GRID):
    g = StateGraph(State)
    g.add_node("global_agent", global_node)
    g.add_node("aggregator", aggregator_node)
    g.add_edge(START, "global_agent")
    for r in range(grid):
        for c in range(grid):
            name = f"tile_agent_r{r}c{c}"
            g.add_node(name, make_tile_node(f"r{r}c{c}"))
            g.add_edge("global_agent", name)   # fan-out: all tile agents run in parallel
            g.add_edge(name, "aggregator")     # fan-in: aggregator waits for every tile agent
    g.add_edge("aggregator", END)
    log.debug("graph: 1 global + %d tile + 1 aggregator nodes", grid * grid)
    return g.compile()
