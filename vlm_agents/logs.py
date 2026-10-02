"""Logging for the agent pipeline.

Every record is tagged with the agent that emitted it (global, tile r1c2,
aggregator, model, mcp), because the tile agents run on parallel threads and
their steps interleave in the output.
"""
import contextlib
import logging
import sys
import time

FORMAT = "%(asctime)s %(levelname)-5s %(agent)-12s %(message)s"
ROOT = "vlm_agents"


class _DefaultAgent(logging.Filter):
    """Records logged without an agent label fall back to the module name."""

    def filter(self, record):
        if not hasattr(record, "agent"):
            record.agent = record.name.rsplit(".", 1)[-1]
        return True


def setup(level="INFO", logfile=None):
    root = logging.getLogger(ROOT)
    root.setLevel(level.upper() if isinstance(level, str) else level)
    root.handlers.clear()
    root.propagate = False

    handlers = [logging.StreamHandler(sys.stderr)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="w"))
    for h in handlers:
        h.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
        h.addFilter(_DefaultAgent())
        root.addHandler(h)
    return root


def get(agent: str):
    """A logger whose records carry `agent` as their label."""
    return logging.LoggerAdapter(logging.getLogger(ROOT), {"agent": agent})


def short(text, limit=160):
    """One-line preview of a model reply, for INFO-level logs."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


@contextlib.contextmanager
def timed(log, what, level=logging.INFO):
    """Log the start and the elapsed time of a step."""
    log.log(level, "%s ...", what)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as e:
        log.error("%s FAILED after %.1fs: %s", what, time.perf_counter() - t0, e)
        raise
    else:
        log.log(level, "%s done in %.1fs", what, time.perf_counter() - t0)
