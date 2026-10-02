import os
import random
from collections import Counter
from PIL import Image

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "datasets", "roco-dataset", "all_data")
SPLITS = ["train", "validation", "test"]
CATEGORIES = ["radiology", "non-radiology"]
SAMPLE_SIZE = 200  # images sampled per split/category, set to None to check every image


def iter_image_paths(images_dir):
    for fname in os.listdir(images_dir):
        yield os.path.join(images_dir, fname)


def check_sizes():
    sizes = Counter()
    total_checked = 0

    for split in SPLITS:
        for category in CATEGORIES:
            images_dir = os.path.join(DATASET_ROOT, split, category, "images")
            if not os.path.isdir(images_dir):
                continue

            paths = list(iter_image_paths(images_dir))
            if SAMPLE_SIZE is not None and len(paths) > SAMPLE_SIZE:
                paths = random.sample(paths, SAMPLE_SIZE)

            for path in paths:
                try:
                    with Image.open(path) as img:
                        sizes[img.size] += 1
                        total_checked += 1
                except Exception as e:
                    print(f"Failed to read {path}: {e}")

    print(f"Checked {total_checked} images\n")
    print(f"{'width x height':<20}{'count':<10}")
    for (w, h), count in sizes.most_common():
        print(f"{w}x{h:<15}{count}")

    if sizes:
        widths = [w for w, h in sizes.elements()]
        heights = [h for w, h in sizes.elements()]
        print(f"\nWidth  min/max/avg: {min(widths)}/{max(widths)}/{sum(widths) / len(widths):.1f}")
        print(f"Height min/max/avg: {min(heights)}/{max(heights)}/{sum(heights) / len(heights):.1f}")


if __name__ == "__main__":
    check_sizes()
