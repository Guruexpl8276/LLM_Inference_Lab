# API Reference

Base URL: `http://localhost:8000` (configurable via `API_HOST` + `API_PORT`).
Auto-generated OpenAPI/Swagger UI: **`/docs`**.

## Inference

### `POST /chat`
Server-Sent Events streaming inference.

**Request**
```json
{
  "prompt": "Explain quantization",
  "model": "llama3.2:3b",
  "max_tokens": 512,
  "temperature": 0.7,
  "system": null
}
```

**Response** — `text/event-stream`. Each frame is `data: {json}\n\n`:
```json
{"token": "Quant", "ttft_ms": 412.3, "done": false, "queue_wait_ms": 0.1, "scheduler_state_at_acquire": "NORMAL"}
{"token": "ization", "ttft_ms": null, "done": false}
...
{"token": "", "ttft_ms": null, "done": true, "stats": {"ttft_ms": 412.3, "tpot_ms": 31.4, "tokens_per_second": 31.8, "output_tokens": 287, "input_tokens": 7, "total_time_s": 9.1}}
```

### `POST /generate`
Non-streaming inference; returns the full response in one JSON.

```json
{
  "model": "llama3.2:3b",
  "response": "...",
  "metrics": { "ttft_ms": 412.3, "tpot_ms": 31.4, "tokens_per_second": 31.8, ... }
}
```

## Telemetry

### `GET /health`
Combined system snapshot: API + Ollama + queue + GPU.

### `GET /gpu`
Live GPU stats (used by dashboard 2.5s poller).

### `GET /queue/status`
SJF scheduler state.

```json
{
  "active_inference": 2,
  "waiting": 1,
  "max_concurrent": 6,
  "queue_capacity_pct": 50.0,
  "backpressure_state": "THROTTLING",
  "scheduler": "SJF",
  "max_seen_depth": 5,
  "vram_used_pct": 67.2,
  "soft_limit_pct": 50.0,
  "hard_limit_pct": 80.0,
  "vram_oom_threshold_pct": 90.0
}
```

### `GET /models`
List of locally available Ollama models (with VRAM size).

## Benchmark

### `POST /benchmark`
Start a background benchmark of a single model. Poll `/benchmark/status` for results.

```json
{ "model": "llama3.2:3b", "prompt_type": "medium", "runs": 3 }
```

### `POST /compare`
Multi-model side-by-side benchmark. `models: null` auto-detects GPU-safe installed models.

```json
{ "models": null, "prompt_type": "medium", "runs": 2 }
```

### `GET /benchmark/status`
Poll for the in-flight benchmark / comparison result.

## Simulation

### `POST /simulate/vram`
VRAM footprint calculator. Formula: `Memory(GB) = P × (Q/8) × (1 + Overhead)`.

```json
{ "params_billions": 3, "bits": 4, "overhead": 0.2 }
```
→
```json
{
  "estimated_vram_gb": 1.8,
  "fits_in_gpu": true,
  "fits_in_free_vram": true,
  "recommendation": "Fits with headroom",
  "formula": "3B × (4/8) × (1 + 0.2) = 1.80 GB"
}
```

## Research log

All runs are persisted to SQLite (`./inference_lab.db` by default).

### `GET /history/runs?model=&limit=`
Most-recent inference runs (one row per `/chat` / `/generate` / `/benchmark` invocation).

### `GET /history/aggregate`
Per-model averages across the entire dataset (TTFT, TPOT, TPS, queue wait, VRAM).

### `GET /history/timeline?model=&limit=`
Chronological points for plotting throughput over time.

### `GET /history/loadtests?limit=`
Previous load-test summaries.

## Stress test

### `POST /loadtest`
Fire `concurrency` parallel `/generate` requests and observe the queue.

```json
{ "model": "llama3.2:3b", "concurrency": 12, "prompt": "...", "max_tokens": 64 }
```

Returns successful/429/error counts, p50/p95 latencies, and a sampled timeline of
queue capacity %, VRAM %, and GPU utilization % through the test window.

## Error responses

| Code | Meaning |
|---|---|
| `429` | Queue saturated or VRAM > 90 %. Includes `Retry-After` header and a structured `detail` body. |
| `409` | A benchmark is already running. |
| `500` | Inference engine error (typically Ollama unreachable or model missing). |
