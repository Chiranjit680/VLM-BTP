"""Per-run artifact store.

Every run writes a directory holding the tiles it cut, every image a tool
produced, and the full reasoning of each agent (the raw model reply at each
step, not just its final report). The tile agents run on parallel threads, so
writes are guarded by a lock.
"""
import json
import os
import re
import threading
from datetime import datetime

from . import config, logs

log = logs.get("store")
_current = None


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


class RunStore:
    def __init__(self, root, image_path, question, reference=""):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        self.dir = os.path.join(root, f"{datetime.now():%Y%m%d-%H%M%S}_{stem}")
        self.images_dir = os.path.join(self.dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.agents = {}
        self.meta = {
            "image": os.path.abspath(image_path),
            "question": question,
            "reference_caption": reference,
            "started": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "tile_grid": config.TILE_GRID,
                "tile_overlap": config.TILE_OVERLAP,
                "global_max_side": config.GLOBAL_MAX_SIDE,
                "max_tool_steps": config.MAX_TOOL_STEPS,
                "max_new_tokens": config.MAX_NEW_TOKENS,
            },
        }
        log.info("saving run artifacts to %s", self.dir)

    def save_image(self, name, image):
        """Write an image and return its path relative to the run directory."""
        rel = os.path.join("images", f"{_slug(name)}.png")
        with self._lock:
            image.save(os.path.join(self.dir, rel))
        log.debug("saved %s (%dx%d)", rel, *image.size)
        return rel

    def record_agent(self, agent, report, steps):
        with self._lock:
            self.agents[agent] = {"report": report, "steps": steps}

    def finish(self, answer, seconds):
        self.meta.update(answer=answer, seconds=round(seconds, 1), agents=self.agents)
        with open(os.path.join(self.dir, "run.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)
        self._write_report(answer, seconds)
        log.info("run artifacts written to %s", self.dir)
        return self.dir

    def _write_report(self, answer, seconds):
        m = self.meta
        out = [f"# Run {os.path.basename(self.dir)}", "",
               f"- **Image**: `{m['image']}`", f"- **Question**: {m['question']}",
               f"- **Duration**: {seconds:.1f}s", "",
               "## Final answer", "", answer, "",
               f"## Ground truth ({m.get('roco_id') or 'no id'})", "",
               m["reference_caption"] or "_not in the ROCO CSVs_", ""]

        for name in sorted(self.agents, key=lambda n: (n != "global", n)):
            a = self.agents[name]
            out += [f"## Agent: {name}", ""]
            for s in a["steps"]:
                out.append(f"**Step {s['step']}** ({s['seconds']:.1f}s) raw reply:")
                out += ["", "```", s["reply"].strip(), "```", ""]
                if s.get("tool"):
                    detail = s.get("error") or f"-> `{s.get('result_image', '')}`"
                    out.append(f"Tool `{s['tool']}({s['args']})` {detail}")
                    out.append("")
            out += ["**Report**:", "", a["report"].strip(), ""]
        with open(os.path.join(self.dir, "report.md"), "w") as fh:
            fh.write("\n".join(out))


class NullStore:
    """Used when the pipeline runs without a store (library use, tests)."""

    dir = None

    def save_image(self, name, image):
        return None

    def record_agent(self, agent, report, steps):
        pass

    def finish(self, answer, seconds):
        return None


def start(root, image_path, question, reference=""):
    global _current
    _current = RunStore(root, image_path, question, reference)
    return _current


def current():
    return _current if _current is not None else NullStore()
