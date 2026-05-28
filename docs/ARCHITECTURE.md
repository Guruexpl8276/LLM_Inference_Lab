# Architecture

LLM Inference Lab implements the research-plan vision *"Architecting an Adaptive LLM Inference System"* as a small, hackable codebase you can read end-to-end in an afternoon.

## High-level layout

```
                          ┌──────────────────────────────────────┐
                          │  Dashboard (frontend/)               │
                          │  Chat · Benchmark · Compare ·        │
                          │  Research Log · Stress Test          │
                          └────────────────┬─────────────────────┘
                                           │  HTTP + SSE
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FastAPI app  (backend/main.py)                                          │
│                                                                          │
│  ┌──────────────────────┐   ┌──────────────────────┐                     │
│  │  SJF scheduler       │   │  Persistent log      │                     │
│  │  (scheduler.py)      │   │  (db.py · SQLite)    │                     │
│  │   - priority queue   │   │   - inference_runs   │                     │
│  │   - backpressure     │   │   - load_tests       │                     │
│  │   - VRAM circuit-    │   └──────────────────────┘                     │
│  │     breaker          │                                                │
│  └──────────┬───────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌──────────────────────┐   ┌──────────────────────┐                     │
│  │  Inference engine    │   │  GPU monitor         │                     │
│  │  (inference.py)      │   │  (gpu_monitor.py)    │                     │
│  │   - streams Ollama   │   │   - nvidia-ml-py     │                     │
│  │   - TTFT / TPOT /    │   │   - nvidia-smi       │                     │
│  │     TPS metrics      │   │     fallback         │                     │
│  └──────────┬───────────┘   └──────────────────────┘                     │
│             │                                                            │
└─────────────┼────────────────────────────────────────────────────────────┘
              │  HTTP (localhost:11434)
              ▼
        ┌───────────┐
        │  Ollama   │  ← llama.cpp under the hood; one process per GPU
        └───────────┘
```

## Core design choices

**Why Ollama and not raw `transformers`?**
On consumer GPUs (4–8 GB), llama.cpp's GGUF Q4 quantization fits 3B models in ~2 GB. The same model loaded via `transformers` at FP16 needs ~6 GB. Ollama gives us the most VRAM-efficient runtime "for free" while exposing a clean HTTP API we can stream from.

**Why SJF over FIFO?**
The research plan calls for demonstrating queueing-theory concepts. FIFO is trivial; Shortest-Job-First is the textbook example of how to reduce mean latency under contention. We approximate "job size" with `max_tokens` (output dominates wall-clock), and break priority ties with FIFO insertion order.

**Why three-tier backpressure (NORMAL / THROTTLING / SHEDDING)?**
Real production systems don't have a single "queue full" line — they smoothly degrade. We follow the same shape:
- `0–50 %` → admit immediately
- `50–80 %` → admit but inject a proportional `asyncio.sleep` delay (intake throttling)
- `> 80 %` or VRAM `> 90 %` → reject with HTTP 429 + `Retry-After` (load shedding)

**Why SQLite for the research log?**
- One file, no daemon, queryable from any tool
- `aiosqlite` keeps the FastAPI event loop unblocked
- Schema denormalized for fast dashboard reads (one row per run)
- Same DB stores both inference runs and load tests so longitudinal analysis is one JOIN away

## Request lifecycle (a `/chat` SSE call)

1. Client sends `{prompt, model, max_tokens, ...}`.
2. `scheduler.acquire()` is called with `estimate_job_tokens(prompt, max_tokens)`.
   - If VRAM > 90 % → raise `HTTPException(429)`.
   - If queue > 80 % capacity → raise `HTTPException(429)` + `Retry-After`.
   - If 50–80 % → `await asyncio.sleep(...)` proportional delay.
   - Else → claim slot (or wait in priority queue for one).
3. SSE generator streams tokens from `stream_generate()` (Ollama `/api/generate`).
4. First chunk carries `ttft_ms`, `queue_wait_ms`, `scheduler_state_at_acquire`.
5. Final chunk carries full `stats` (TPOT, TPS, output_tokens, total_time_s).
6. `finally:` releases the slot, logs the run to SQLite, wakes the next-smallest waiter.

## File-by-file

| File | Role |
|---|---|
| `backend/main.py` | FastAPI app, routes, request models, lifespan/DB init |
| `backend/inference.py` | Ollama HTTP streaming + perf metric extraction |
| `backend/scheduler.py` | SJF priority queue + backpressure |
| `backend/db.py` | Async SQLite log (inference_runs, load_tests) |
| `backend/benchmark.py` | Multi-run benchmark engine + compare harness |
| `backend/gpu_monitor.py` | NVIDIA telemetry via nvidia-ml-py or nvidia-smi |
| `frontend/index.html` | Dashboard shell + 5 tabs |
| `frontend/app.js` | SSE chat, Chart.js viz, history/loadtest UI |
| `frontend/style.css` | Dark research-lab theme |

## Where to extend next

- **Speculative decoding** — Ollama supports `draft_model`; add a comparison endpoint
- **Multi-turn chat with KV-cache reuse** — measure TPOT degradation as context grows
- **Cold vs warm TTFT** — call `ollama stop <model>` then time the first request
- **Energy/token metric** — multiply `power_draw_w` by `total_time_s`, divide by `output_tokens`
