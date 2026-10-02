import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "qwen2.5-vl-3b-instruct")

GLOBAL_MAX_SIDE = 768      # longest side of the image shown to the global agent
TILE_GRID = 3              # tiles per side -> 3x3 = 9 tiles, one tile agent each
TILE_OVERLAP = 0.15        # fraction of a tile overlapped with its neighbours
MAX_TOOL_STEPS = 3         # tool calls allowed per agent before it must answer
MAX_NEW_TOKENS = 256
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
