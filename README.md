# LLM Inference Lab

> A research-grade dashboard for **measuring** local LLM inference — not just running it.
> Streams from Ollama, instruments every request with TTFT / TPOT / throughput, runs SJF-scheduled stress tests, and persists every run to a queryable SQLite log.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Backend-Ollama-black.svg)](https://ollama.com/)

---

## What this is

A small, hackable inference observatory you can read end-to-end in an afternoon. Built originally to test the research plan *"Architecting an Adaptive LLM Inference System"* on a 4 GB GTX 1650, but the codebase makes **no assumption about your GPU** — pick the model preset that fits your card and go.

### Features

- **Live chat with TTFT / TPOT / throughput** stamped on every reply (SSE-streamed)
- **Multi-model side-by-side benchmark** with auto-detection of VRAM-safe presets
- **SJF priority queue** with three-tier backpressure (NORMAL → THROTTLING → SHEDDING)
- **Stress test endpoint** that fires N parallel requests and plots queue/VRAM/GPU over time
- **Persistent research log** — every run goes to SQLite for longitudinal analysis
- **VRAM simulator** — compute model footprint with `P × (Q/8) × (1 + overhead)` before pulling
- **Dashboard** — 5 tabs (Chat, Benchmark, Compare, Research Log, Stress Test), live GPU charts
- **Zero cloud** — entirely local; works fully offline once models are pulled

---

## Quick start

### Requirements

- Python 3.10+
- An NVIDIA GPU with recent drivers (optional but strongly recommended — CPU works, just slowly)
- [Ollama](https://ollama.com/download) installed

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/llm-inference-lab.git
cd llm-inference-lab

# Windows
.\setup.ps1

# Linux / macOS
chmod +x setup.sh run.sh
./setup.sh
```

`setup` detects your GPU, recommends a model preset for its VRAM tier, creates `.venv`, installs dependencies, and (with your consent) pulls those models via Ollama.

### 2. Run

```bash
# Windows
.\run.ps1

# Linux / macOS
./run.sh
```

Then open [http://localhost:8000/dashboard](http://localhost:8000/dashboard).

API docs auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Model presets by GPU

The setup script auto-suggests one of these based on your card's VRAM. Override anything you want — every Ollama-supported model works.

| VRAM | Tier | Suggested models | Notes |
|---|---|---|---|
| CPU only | `cpu` | `tinyllama`, `gemma3:1b` | Slow but functional |
| 4 GB | `4GB` | `llama3.2:3b`, `phi3:mini`, `tinyllama` | GTX 1650, GTX 1050 Ti |
| 6 GB | `6GB` | + `qwen2.5:3b`, `gemma2:2b` | GTX 1660, RTX 2060 |
| 8 GB | `8GB` | `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b` | RTX 3060 (8GB), 3070 |
| 12 GB | `12GB` | `llama3.1:8b`, `qwen2.5:14b`, `mistral-nemo` | RTX 3060 (12GB), 4070 |
| 16 GB | `16GB` | `qwen2.5:14b`, full-context 8B models | RTX 4060 Ti, 4070 Ti |
| 24 GB+ | `24GB+` | `qwen2.5:32b`, `llama3.3:70b-q4` | RTX 3090, 4090 |

Pull manually any time:

```bash
ollama pull <model>:<tag>
```

---

## Project layout

```
llm-inference-lab/
├── backend/
│   ├── main.py          ← FastAPI app, routes, lifespan
│   ├── inference.py     ← Ollama streaming + perf metrics
│   ├── scheduler.py     ← SJF priority queue + backpressure
│   ├── db.py            ← Async SQLite research log
│   ├── benchmark.py     ← Multi-run benchmark + compare
│   └── gpu_monitor.py   ← NVIDIA telemetry (nvidia-ml-py / nvidia-smi)
├── frontend/
│   ├── index.html       ← 5-tab dashboard
│   ├── app.js           ← SSE chat, Chart.js viz, history/loadtest UI
│   └── style.css        ← Dark research-lab theme
├── docs/
│   ├── ARCHITECTURE.md  ← How it all fits together
│   ├── API.md           ← Full endpoint reference
│   └── PERSONAL-H-DRIVE-SETUP.md  ← Running entirely off a non-system drive
├── app_gradio.py        ← Optional Gradio UI (for HF Spaces deployment)
├── run.ps1 / run.sh     ← Cross-platform launchers
├── setup.ps1 / setup.sh ← Guided first-time installer
├── requirements.txt
├── .env.example
└── LICENSE
```

---

## Configuration

Everything is optional. Copy `.env.example` to `.env` and edit any of:

| Variable | Default | Purpose |
|---|---|---|
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Where the FastAPI server binds |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon URL (point to a remote box if you like) |
| `DEFAULT_MODEL` | `llama3.2:3b` | Default model for `/chat` / `/generate` when omitted |
| `OLLAMA_NUM_GPU` | `99` | Transformer layers offloaded to GPU (99 = all) |
| `OLLAMA_NUM_CTX` | `2048` | Per-request context window |
| `MAX_CONCURRENT` | `6` | SJF queue hard ceiling |
| `SOFT_LIMIT_PCT` | `0.50` | Throttling threshold (intake delay starts here) |
| `HARD_LIMIT_PCT` | `0.80` | Load-shedding threshold (HTTP 429 starts here) |
| `VRAM_OOM_THRESHOLD_PCT` | `90.0` | VRAM % above which all new requests get 429 |
| `INFERENCE_DB_PATH` | `./inference_lab.db` | Where to store the persistent run log |
| `OLLAMA_MODELS` | (Ollama default) | Override Ollama's model storage location |
| `VENV_PATH` | `./.venv` | Override venv location (e.g. on a different drive) |

---

## API surface

5 inference / telemetry endpoints, 4 history endpoints, 1 stress-test endpoint, 1 simulator. See [docs/API.md](docs/API.md) for the full reference. A taste:

```bash
# Stream a chat response with TTFT / TPOT / TPS metrics
curl -N -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV cache in two sentences","model":"llama3.2:3b"}'

# Fire 12 parallel requests, watch the queue saturate
curl -X POST http://localhost:8000/loadtest \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","concurrency":12,"max_tokens":64}'

# Aggregate stats across every run ever logged
curl http://localhost:8000/history/aggregate
```

---

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture. The short version:

```
Browser ──HTTP/SSE──> FastAPI ──> SJF scheduler ──> Inference engine ──HTTP──> Ollama
                          │             │
                          │             └──> backpressure (NORMAL/THROTTLING/SHEDDING)
                          │
                          ├──> SQLite log (every run + every load test)
                          └──> GPU monitor (nvidia-ml-py)
```

---

## Hacking on it

The codebase is small on purpose. The most interesting files to read first:

1. `backend/scheduler.py` — the SJF queue with backpressure (~150 lines)
2. `backend/main.py` — the FastAPI app wiring everything together
3. `backend/inference.py` — Ollama HTTP streaming + metric extraction
4. `frontend/app.js` — dashboard logic, including the stress-test UI

Good first extensions to try:

- **Cold-vs-warm TTFT**: time the first request after `ollama stop <model>`
- **Speculative decoding**: Ollama supports `draft_model`; benchmark it
- **Energy per token**: `power_draw_w × total_time_s / output_tokens` from the GPU stats already logged
- **CSV export** from the Research Log tab for pandas analysis

---

## Why "Lab"?

This is a tool for *understanding* inference, not a production serving stack. If you want production, look at vLLM / TGI / TensorRT-LLM. If you want to *see* what TTFT looks like when your queue fills up and your VRAM hits 90 % — you're in the right place.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on top of [Ollama](https://ollama.com/) (llama.cpp), [FastAPI](https://fastapi.tiangolo.com/), [Chart.js](https://www.chartjs.org/), and [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/).
