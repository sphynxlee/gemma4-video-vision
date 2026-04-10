# Gemma 4 + YOLO Real-Time Video Vision

Real-time video analysis pipeline that combines **YOLO instance segmentation** with **Gemma 4 multimodal scene captioning**, running entirely on local hardware — no cloud APIs required.

Inspired by [@MaziyarPanahi's demo](https://x.com/MaziyarPanahi/status/2040067043320307937) of Gemma 4 + SAM3 on an M2 MacBook.

## What It Does

| Component | Role | Output |
|-----------|------|--------|
| **YOLO** (Ultralytics) | Object detection & segmentation | Colored masks and bounding boxes on each detected object |
| **Gemma 4** (via LM Studio) | Scene understanding | Natural-language caption displayed at the bottom of the frame |

- **YOLO** runs every frame for real-time visual annotation
- **Gemma 4** runs every few seconds in a background thread for scene descriptions
- Both models are independent — YOLO works fine without LM Studio; Gemma 4 adds open-ended understanding on top

### Why Two Models?

YOLO can only classify objects from 80 fixed COCO categories. If an object is not in that list (e.g. a toy tank), it will guess wrong. Gemma 4 has no such limitation — it truly "understands" the scene and can describe anything it sees, including reading text, identifying context, and describing actions.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) or a machine with a decent GPU
- **Python 3.10+**
- **LM Studio** — [download here](https://lmstudio.ai)

## Setup

### 1. Clone and Create Virtual Environment

```bash
git clone <your-repo-url>
cd gemma4-video-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up LM Studio

1. Open LM Studio and search for `gemma-4` (use `Cmd+Shift+M`)
2. Download a Gemma 4 model — recommended options:
   | Model | Size (Q4) | Best For |
   |-------|-----------|----------|
   | `google/gemma-4-e4b` | ~3 GB | Low memory / fast response |
   | `google/gemma-4-26b-a4b` | ~15 GB | Good balance (MoE, only 4B active) |
   | `google/gemma-4-31b` | ~17 GB | Maximum accuracy |
3. Go to the **Developer** tab, load the model (`Cmd+L`)
4. **Important**: Click the **Load** tab (not Info) on the right panel and set **GPU Offload** to maximum for better performance
5. Start the server — it should show **Running** on port `1234`

### 3. Run

```bash
# Webcam (default)
python main.py

# Video file
python main.py --source path/to/video.mp4

# All options
python main.py --help
```

Press `q` to quit the video window.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | `0` for webcam, or path to a video file |
| `--yolo-model` | `yolo11n-seg.pt` | YOLO model variant (auto-downloaded on first use) |
| `--gemma-model` | `google/gemma-4-26b-a4b` | Model name as shown in LM Studio |
| `--lm-studio-url` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `--caption-interval` | `3.0` | Seconds between caption updates |
| `--conf` | `0.35` | YOLO confidence threshold |

### YOLO Model Variants

Larger models are more accurate but slower. First use auto-downloads the weights.

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolo11n-seg.pt` | ~6 MB | Fastest | Fair |
| `yolo11s-seg.pt` | ~12 MB | Fast | Good |
| `yolo11m-seg.pt` | ~42 MB | Medium | Better |
| `yolo11l-seg.pt` | ~50 MB | Slower | Best |

## Troubleshooting

### Caption shows `[caption error: Connection error.]`

LM Studio server is not running. Open LM Studio → Developer tab → make sure the server status shows **Running**.

### Caption stays at "Waiting for first caption..."

The model name in the script doesn't match LM Studio. Check the exact name in LM Studio's Developer tab and pass it via `--gemma-model`.

### Gemma 4 returns empty responses

Gemma 4 is a **thinking model** — it uses internal reasoning tokens before producing the final answer. If `max_tokens` is too low, the model exhausts its budget on thinking and returns empty content. The script sets `max_tokens=512` to handle this. If you still see issues, the model may need more tokens for complex scenes.

### YOLO misidentifies objects

YOLO only knows 80 COCO categories. Objects outside this set (tanks, specific tools, unusual items) will be misclassified. This is by design — Gemma 4's caption provides the accurate description.

## Architecture

```
Video source (webcam / file)
    │
    ├──→ Every frame ──→ YOLO ──→ Segmentation masks + labels
    │
    ├──→ Every N seconds ──→ Encode JPEG ──→ base64
    │       │
    │       └──→ HTTP POST to localhost:1234 (LM Studio)
    │               └──→ Gemma 4 inference ──→ Scene description
    │
    └──→ Composite: annotated frame + caption bar + FPS counter
            └──→ Display window
```

The connection between Python and Gemma 4 is a plain **HTTP request** — LM Studio exposes an OpenAI-compatible REST API at `localhost:1234`. The script uses the `openai` Python package with `base_url` pointed to the local server.

## Hardware Notes

- Apple Silicon's **unified memory** lets CPU and GPU share the same RAM — no separate VRAM needed
- 4-bit quantization (Q4) reduces the 26B model from ~52 GB to ~15 GB
- 16 GB RAM minimum for smaller models; 32 GB recommended for the 26B variant
- Enable **GPU Offload** in LM Studio's Load tab for significantly faster inference via Metal

## License

MIT
