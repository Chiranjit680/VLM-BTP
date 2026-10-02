"""Image-transformation tools available to agents.

Every tool takes a PIL image plus keyword args and returns a new PIL image.
Crop coordinates are normalised to [0, 1] so they are independent of resolution.
"""
from PIL import Image, ImageFilter, ImageOps

TOOLS = {}


def tool(name, signature, doc):
    def deco(fn):
        TOOLS[name] = {"fn": fn, "signature": signature, "doc": doc}
        return fn
    return deco


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


@tool("zoom", "zoom(x0, y0, x1, y1)", "Crop a normalised box (0-1) and upscale it to inspect fine detail.")
def zoom(img, x0, y0, x1, y1):
    w, h = img.size
    x0, x1 = sorted((_clamp(x0, 0, 1), _clamp(x1, 0, 1)))
    y0, y1 = sorted((_clamp(y0, 0, 1), _clamp(y1, 0, 1)))
    box = (int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 8), max(int(y1 * h), int(y0 * h) + 8))
    crop = img.crop(box)
    scale = max(1.0, 512 / max(crop.size))
    return crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)


@tool("autocontrast", "autocontrast(cutoff)", "Stretch intensities; cutoff (0-20) is the percent clipped at each end.")
def autocontrast(img, cutoff=1):
    return ImageOps.autocontrast(img, cutoff=_clamp(cutoff, 0, 20))


@tool("equalize", "equalize()", "Histogram equalisation to reveal low-contrast structures.")
def equalize(img):
    return ImageOps.equalize(img)


@tool("gamma", "gamma(value)", "Gamma correction; value <1 brightens shadows, >1 darkens (0.3-3).")
def gamma(img, value=0.7):
    g = _clamp(value, 0.3, 3.0)
    lut = [int(255 * (i / 255) ** g) for i in range(256)]
    return img.point(lut * len(img.getbands()))


@tool("sharpen", "sharpen()", "Unsharp mask to emphasise edges and boundaries.")
def sharpen(img):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


@tool("denoise", "denoise()", "Median filter to suppress speckle noise.")
def denoise(img):
    return img.filter(ImageFilter.MedianFilter(3))


@tool("edges", "edges()", "Edge map of the image, useful for outlines and shapes.")
def edges(img):
    return ImageOps.autocontrast(img.convert("L").filter(ImageFilter.FIND_EDGES)).convert("RGB")


@tool("invert", "invert()", "Invert intensities (can make bright-on-dark findings easier to see).")
def invert(img):
    return ImageOps.invert(img.convert("RGB"))


def describe_tools(names=None):
    names = names or list(TOOLS)
    return "\n".join(f"- {TOOLS[n]['signature']}: {TOOLS[n]['doc']}" for n in names)


def apply_tool(name, img, args):
    """Run a tool; raises ValueError with a readable message on bad calls."""
    if name not in TOOLS:
        raise ValueError(f"unknown tool '{name}'")
    try:
        return TOOLS[name]["fn"](img.convert("RGB"), **(args or {}))
    except TypeError as e:
        raise ValueError(f"bad arguments for {name}: {e}")
