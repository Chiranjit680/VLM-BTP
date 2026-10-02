"""Run the multi-agent pipeline on one image from the ROCO dataset (or any file).

  python run.py -n 0 --cuda 6
  python run.py -n 12 --cuda 3 -q "Is there any abnormality, and where?"
  python run.py --image path/to/img.jpg --cuda 6
"""
import argparse
import csv
import glob
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--index", type=int, default=0, help="nth image of the split, in ROCO id order")
ap.add_argument("--id", default=None, help="pick by ROCO id, e.g. ROCO_00001 (or just 1)")
ap.add_argument("--cuda", default=None, help="GPU id to run on, e.g. 6")
ap.add_argument("--split", default="test", choices=["test", "train", "validation"])
ap.add_argument("--category", default="radiology", choices=["radiology", "non-radiology"])
ap.add_argument("--image", default=None, help="explicit image path; overrides -n/--split/--category")
ap.add_argument("-q", "--question", default=None)
ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                help="DEBUG also logs prompts, raw model replies and token counts")
ap.add_argument("--log-file", default=None, help="also write the log to this file")
ap.add_argument("--out-dir", default="runs", help="where each run's tiles and reasoning are saved")
ap.add_argument("--no-save", action="store_true", help="do not save run artifacts")
args = ap.parse_args()

# Must be set before torch is imported anywhere below. PCI_BUS_ID makes --cuda N
# mean the same card as `nvidia-smi` index N (CUDA's default order does not).
if args.cuda is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from vlm_agents import logs  # noqa: E402

logs.setup(args.log_level, args.log_file)

from PIL import Image  # noqa: E402

from vlm_agents import config, store  # noqa: E402
from vlm_agents.graph import DEFAULT_QUESTION, build_graph  # noqa: E402

DATA_ROOT = os.path.join(config.ROOT, "datasets", "roco-dataset", "all_data")


def lookup_ground_truth(category_dir, name):
    """Return (roco_id, caption) for an image, from that folder's CSV."""
    for path in glob.glob(os.path.join(category_dir, "*.csv")):  # testdata.csv / valdata.csv / ...
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("name") == name:
                    return row.get("id", ""), (row.get("caption") or "").strip()
    return "", ""


def dataset_rows(split, category):
    """CSV rows for a split in ROCO id order, skipping ids whose image file is missing."""
    d = os.path.join(DATA_ROOT, split, category)
    images_dir = os.path.join(d, "images")
    on_disk = set(os.listdir(images_dir))
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
        with open(path, newline="") as fh:
            rows += [r for r in csv.DictReader(fh) if r.get("name") in on_disk]
    rows.sort(key=lambda r: r["id"])
    return images_dir, rows


def normalise_id(value):
    """Accept ROCO_00001, roco_00001 or a bare 1."""
    value = str(value).strip()
    return f"ROCO_{int(value):05d}" if value.isdigit() else value.upper()


def pick_image(split, category, index=None, roco_id=None):
    """Return (path, roco_id, caption) for a ROCO id, or for the nth id of a split."""
    images_dir, rows = dataset_rows(split, category)
    if roco_id:
        wanted = normalise_id(roco_id)
        row = next((r for r in rows if r["id"] == wanted), None)
        if row is None:
            raise SystemExit(f"{wanted} not found in {split}/{category} "
                             f"(ids run {rows[0]['id']}..{rows[-1]['id']})")
    else:
        if not -len(rows) <= index < len(rows):
            raise SystemExit(f"index {index} out of range: {split}/{category} has {len(rows)} images")
        row = rows[index]
    return os.path.join(images_dir, row["name"]), row["id"], row["caption"].strip()


if args.image:
    image_path = os.path.abspath(args.image)
    # an explicit path inside the dataset still has a caption two levels up
    roco_id, reference = lookup_ground_truth(os.path.dirname(os.path.dirname(image_path)),
                                             os.path.basename(image_path))
else:
    image_path, roco_id, reference = pick_image(args.split, args.category, args.index, args.id)

question = args.question or DEFAULT_QUESTION
image_path = os.path.abspath(image_path)

run_log = logs.get("run")
with Image.open(image_path) as _im:
    run_log.info("image: %s (%dx%d, %s)", image_path, *_im.size, _im.mode)
run_log.info("question: %s", question)
if reference:
    run_log.info("ground truth (%s): %s", roco_id or "unknown id", logs.short(reference))
else:
    run_log.warning("no ground-truth caption found for this image")

ground_truth = f"== GROUND TRUTH ({roco_id or 'no id'}) ==\n {reference}" if reference else \
    "== GROUND TRUTH ==\n (none: image is not in the ROCO CSVs)"

print(f"== IMAGE ==\n {image_path}")
print(ground_truth)
print(f"== QUESTION ==\n {question}\n")

if not args.no_save:
    run_store = store.start(
        os.path.join(config.ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir,
        image_path, question, reference)
    run_store.meta["roco_id"] = roco_id

t0 = time.perf_counter()
result = build_graph().invoke({"image": Image.open(image_path), "question": question})
elapsed = time.perf_counter() - t0
run_log.info("pipeline finished in %.1fs", elapsed)
run_dir = store.current().finish(result["answer"], elapsed)

def tools_used(steps):
    return json.dumps([{k: s[k] for k in ("tool", "args", "error") if k in s}
                       for s in steps if "tool" in s])


print("== GLOBAL ==\n", result["global_summary"], "\n  tools:", tools_used(result["global_trace"]))
for r in sorted(result["tile_reports"], key=lambda r: r["tile_id"]):
    print(f"== TILE {r['tile_id']} ==\n", r["report"], "\n  tools:", tools_used(r["trace"]))
print(ground_truth)
print("== ANSWER ==\n", result["answer"])
if run_dir:
    print(f"== SAVED ==\n {run_dir}")
