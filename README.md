# Multi-agent VLM reasoning over biomedical images

A LangGraph pipeline that reasons about a biomedical image with **three kinds of agent**, all
driven by one local **Qwen2.5-VL-3B-Instruct**. The idea: a 3B model cannot resolve fine detail
in a whole downscaled image, so one agent takes the global view for context while nine agents
each study one tile at full resolution, and a third agent merges their findings.

Image-transformation tools (zoom, contrast, edges, …) are not called in-process — they live in a
separate **MCP server**, so any MCP client can use them and agents share one stateless tool service.

---

## Quick start

The project runs in the `env` conda environment. The `base` environment has no torch.

```bash
conda activate env
cd /home/btech/2023/chiranjit.saha23b/VLM

python run.py -n 0 --cuda 6
```

That runs the first ROCO test image on GPU 6 and prints the global summary, all nine tile
reports, the ground-truth caption and the final answer. Takes about 80 seconds.

### More ways to pick an image

```bash
python run.py --id ROCO_59505 --cuda 6          # by ROCO id (or just --id 59505)
python run.py -n 12 --cuda 4                    # 12th image in ROCO id order
python run.py -n 0 --split validation --cuda 6  # another split
python run.py --category non-radiology --cuda 6 # another category
python run.py --image path/to/scan.png --cuda 6 # any file on disk
```

### Asking a question

Without `-q` the pipeline describes and interprets the image. With it, every agent is steered
by the question:

```bash
python run.py -n 0 --cuda 6 -q "Is there any abnormality, and where?"
```

### Logging and artifacts

```bash
python run.py -n 0 --cuda 6 --log-level DEBUG        # + prompts, raw replies, tok/s
python run.py -n 0 --cuda 6 --log-file runs/n0.log   # tee the log to a file
python run.py -n 0 --cuda 6 --out-dir experiments    # save artifacts elsewhere
python run.py -n 0 --cuda 6 --no-save                # save nothing
python run.py -n 0 --cuda 6 2>run.log                # logs to stderr, results to stdout
```

### Picking a GPU

`--cuda N` matches `nvidia-smi` numbering (the code sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`; CUDA's
default order does not match). GPUs 0–2 are V100s, which run fp16; 3–6 are A100s, which run
bf16. Check free memory first — the model needs roughly 8 GB.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

---

## What happens in a run

```
run.py  ──►  pick image, pin GPU, start logging + artifact store
                          │
                          ▼
            ┌────────────────────────────────┐
            │ node: global_agent             │  whole image, downscaled to ≤768 px
            │   ReAct tool loop (≤3 calls)   │  → global_summary
            └───────────────┬────────────────┘
                            │  cuts the full-res image into a 3×3 grid, 15% overlap
      ┌────────┬────────────┼────────────┬────────┐      9 parallel edges
      ▼        ▼            ▼            ▼        ▼
 tile_agent_r0c0   r0c1    r1c1   …    r2c2       (9 separate graph nodes)
 each: its own tile at full res + the question + global_summary, own tool loop
      └────────┴────────────┬────────────┴────────┘
                            │  reports merged by an operator.add reducer
                            ▼
            ┌────────────────────────────────┐
            │ node: aggregator               │  text only, no image
            │   → final answer, cites tiles  │
            └────────────────────────────────┘
                            │
                  runs/<timestamp>_<image>/   tiles, tool images, every raw reply
```

**The tool loop.** Each global and tile agent must answer with exactly one JSON object: either
`{"tool": "...", "args": {...}}` to transform what it is looking at, or `{"final": "..."}` to
report. A tool call replaces its current image and the loop repeats, up to `MAX_TOOL_STEPS`;
the last step forces a final answer. A malformed reply falls back to using the raw text, and a
tool error is fed back as a failed step so the model can recover.

**The tools**, served over MCP: `zoom` (normalised crop plus upscale), `autocontrast`,
`equalize`, `gamma`, `sharpen`, `denoise`, `edges`, `invert`. The menu shown to the model is
generated from the server's `list_tools`, so adding a tool to the server exposes it to every
agent with no prompt change.

---

## Repository map

| File | Role |
|---|---|
| [`run.py`](run.py) | CLI: picks the image, pins the GPU, runs the graph, prints results |
| [`vlm_agents/graph.py`](vlm_agents/graph.py) | LangGraph wiring: 1 global + 9 tile + 1 aggregator node; tile cutting |
| [`vlm_agents/agents.py`](vlm_agents/agents.py) | The three agent roles and the shared ReAct tool loop |
| [`vlm_agents/model.py`](vlm_agents/model.py) | Qwen2.5-VL singleton, dtype selection, cuDNN probe |
| [`vlm_agents/mcp_server.py`](vlm_agents/mcp_server.py) | Stateless MCP server exposing the 8 image tools |
| [`vlm_agents/mcp_client.py`](vlm_agents/mcp_client.py) | Sync facade over the MCP session for the graph nodes |
| [`vlm_agents/tools/image_tools.py`](vlm_agents/tools/image_tools.py) | The PIL implementations the server wraps |
| [`vlm_agents/store.py`](vlm_agents/store.py) | Per-run artifacts: tiles, tool images, reasoning |
| [`vlm_agents/logs.py`](vlm_agents/logs.py) | Logging, tagged per agent |
| [`vlm_agents/config.py`](vlm_agents/config.py) | All tunables |

---

## Configuration

Everything tunable is in [`vlm_agents/config.py`](vlm_agents/config.py):

| Setting | Default | Meaning |
|---|---|---|
| `TILE_GRID` | `3` | Tiles per side. 3 means 9 tile agents — the node count follows this |
| `TILE_OVERLAP` | `0.15` | Neighbour overlap, so a finding on a border is not split |
| `GLOBAL_MAX_SIDE` | `768` | Longest side of the image the global agent sees |
| `MAX_TOOL_STEPS` | `3` | Tool calls an agent may make before it must answer |
| `MAX_NEW_TOKENS` | `256` | Generation cap per agent reply (the aggregator gets 384) |
| `MIN_PIXELS` / `MAX_PIXELS` | `128`/`768` × 28² | Visual-token budget per image |

---

## What a run saves

Every run writes `runs/<timestamp>_<image-stem>/`:

```
run.json     machine-readable: config, timings, ROCO id, ground truth,
             and every agent's steps (raw reply, latency, tool call, result image)
report.md    the same as a readable walkthrough, answer next to ground truth
images/
  source.png  global_input.png
  tile_r0c0.png … tile_r2c2.png     all nine tiles as cut
  tile_r1c1_step1_zoom.png          every image a tool produced
```

So for any agent you can see what it said, what it asked for, and exactly which image it was
looking at when it said it.

---

## Observed behaviour

From four completed runs on the ROCO test split:

- **~80 s per image** end to end (75–83 s), plus 30–60 s of model loading.
- **The tile agents really do run concurrently.** One run spent 502 s of tile-agent time in
  82.8 s of wall clock, roughly a 6× overlap, because torch releases the GIL during CUDA work.
- **Output is deterministic** — decoding is greedy, so the same image twice gives identical text.
- **The JSON protocol holds.** Every agent emitted valid JSON.

### Known weaknesses

1. **The model never uses a tool.** Across four runs and 44 opportunities it always answered
   immediately with `{"final": ...}`. The tool loop and MCP server work (verified directly), but
   a 3B model does not spontaneously reach for them. Fixing this likely means prompting for a
   specific first action rather than offering a menu.
2. **Tile agents describe the whole image, not their tile.** In one run `tile r0c0` and
   `tile r1c1` returned identical text describing the entire abdomen — they echo the
   `global_summary` handed to them as context instead of reporting what is in front of them.
3. **No tile ever reports "nothing relevant"**, so the aggregator receives nine confident and
   largely duplicate reports, which is the main thing dragging on answer quality.
4. **Tiles are a fixed grid**, not regions chosen by the global agent.

---

## Environment notes

This conda env has had CUDA 13 packages installed over its CUDA 12 ones, which broke things in
three ways. All are handled or repaired, but they are worth knowing if the env is rebuilt.

- **cuDNN.** `nvidia-cudnn-cu13` overwrote `nvidia-cudnn-cu12`; both ship `libcudnn.so.9`, so
  torch (cu121) loads the CUDA 13 build and every cuDNN call fails with
  `CUDNN_STATUS_NOT_INITIALIZED`. [`model.py`](vlm_agents/model.py) probes cuDNN once at startup
  and disables it if broken — Qwen2.5-VL only uses it for the vision patch-embed conv3d, so the
  cost is negligible. The proper repair:
  ```bash
  pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70
  ```
- **torchvision.** Its `utils.py` had been replaced with an ancient version, which made every
  `transformers` import fail. Repaired with
  `pip install --force-reinstall --no-deps torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`.
- **MCP 2.x.** `FastMCP` is now `MCPServer`, and result fields are `input_schema` / `is_error`.
  The code targets 2.x; `requirements.txt` pins `mcp>=2`.
- **bf16.** `torch.cuda.is_bf16_supported()` returns `True` on the V100s, where bf16 is emulated
  and slow, so the code tests compute capability ≥ 8.0 instead.

Installed: torch 2.5.1+cu121, transformers 5.3.0, langgraph 1.1.3, mcp 2.2.0.

---

## Adding a tool

Two steps, both in `vlm_agents/`:

1. Implement it in [`tools/image_tools.py`](vlm_agents/tools/image_tools.py) with the `@tool`
   decorator — it takes a PIL image and returns one.
2. Expose it in [`mcp_server.py`](vlm_agents/mcp_server.py) as a `@mcp.tool()` function taking
   `image_b64` plus your arguments, with a one-line docstring.

The docstring becomes the description the agents see. No prompt edits are needed — the tool menu
is built from the server at runtime. Keep crop coordinates normalised to 0–1 so they work at any
resolution.

Run the tool server standalone with `python -m vlm_agents.mcp_server`.
