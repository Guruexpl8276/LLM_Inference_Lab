"""
FastAPI Backend — LLM Inference Lab

Research-grade endpoints:
  Core inference :  /chat (SSE), /generate, /benchmark, /compare
  Telemetry      :  /gpu, /health, /queue/status, /models
  Simulation     :  /simulate/vram
  Research log   :  /history/runs, /history/aggregate, /history/timeline,
                    /history/loadtests
  Stress test    :  /loadtest

Scheduling      :  SJF priority queue (backend.scheduler) with backpressure
Persistence     :  SQLite (default ./inference_lab.db, override with INFERENCE_DB_PATH)
"""

import asyncio
import json
import os
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.inference import (
    stream_generate, generate, list_models, check_ollama_health
)
from backend.benchmark import benchmark_model, compare_models, BENCHMARK_PROMPTS
from backend.gpu_monitor import get_gpu_stats
from backend.scheduler import (
    scheduler, estimate_job_tokens,
    MAX_CONCURRENT, SOFT_LIMIT_PCT, HARD_LIMIT_PCT, VRAM_OOM_THRESHOLD_PCT,
)
from backend import db

load_dotenv()


# ─── Lifespan: init DB once on startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield


app = FastAPI(
    title="LLM Inference Lab API",
    description="Research-grade local LLM inference benchmarking with SJF scheduling, backpressure, and persistent telemetry.",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Request / Response Models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    model: str = Field(default="llama3.2:3b")
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system: Optional[str] = None


class BenchmarkRequest(BaseModel):
    model: str = Field(default="llama3.2:3b")
    prompt_type: str = Field(default="medium")
    runs: int = Field(default=3, ge=1, le=10)


class CompareRequest(BaseModel):
    models: Optional[list] = None
    prompt_type: str = Field(default="medium")
    runs: int = Field(default=2, ge=1, le=5)


class VRAMSimRequest(BaseModel):
    """Memory(GB) = P × (Q/8) × (1 + Overhead)"""
    params_billions: float = Field(..., gt=0)
    bits: int = Field(default=16)
    overhead: float = Field(default=0.20, ge=0.0, le=1.0)


class LoadTestRequest(BaseModel):
    """
    Fire N parallel /generate calls to observe backpressure transitions.
    Demonstrates the research plan's queueing-theory section in action.
    """
    model: str = Field(default="llama3.2:3b")
    concurrency: int = Field(default=10, ge=2, le=50,
                              description="Number of parallel requests to launch")
    prompt: str = Field(default="Write one short sentence about GPUs.")
    max_tokens: int = Field(default=64, ge=8, le=512)


_benchmark_state = {"running": False, "progress": [], "result": None}


# ─── Helper: shared logging path for /chat and /generate ──────────────────────
async def _log_run(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    ttft_ms: Optional[float],
    tpot_ms: Optional[float],
    total_time_s: Optional[float],
    tokens_per_second: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    vram_start_mb: Optional[float],
    queue_wait_ms: Optional[float],
    backpressure_state: Optional[str],
    queue_depth_start: int,
    error: Optional[str] = None,
) -> None:
    gpu_end = get_gpu_stats()
    await db.log_inference_run(
        ts=time.time(),
        endpoint=endpoint,
        model=model,
        prompt_type="adhoc",
        prompt_chars=len(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_time_s=total_time_s,
        tokens_per_second=tokens_per_second,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        vram_used_mb_start=vram_start_mb,
        vram_used_mb_end=gpu_end.vram_used_mb,
        gpu_util_pct_end=gpu_end.gpu_utilization_pct,
        queue_depth_start=queue_depth_start,
        queue_wait_ms=queue_wait_ms,
        backpressure_state=backpressure_state,
        error=error,
    )


# ─── Routes: UI / static ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Dashboard not found</h1><p>frontend/index.html missing.</p>",
        status_code=404,
    )


# ─── Routes: system telemetry ────────────────────────────────────────────────
@app.get("/health")
async def health():
    ollama = await check_ollama_health()
    gpu = get_gpu_stats()
    return {
        "api": "online",
        "ollama": ollama,
        "queue": {
            "active": scheduler.active,
            "waiting": scheduler.waiting,
            "max": MAX_CONCURRENT,
            "capacity_pct": round((scheduler.active + scheduler.waiting) / MAX_CONCURRENT * 100, 1),
            "state": scheduler.current_state(),
        },
        "gpu": {
            "name": gpu.name,
            "vram_used_mb": gpu.vram_used_mb,
            "vram_free_mb": gpu.vram_free_mb,
            "vram_total_mb": gpu.vram_total_mb,
            "vram_used_pct": gpu.vram_used_pct,
            "gpu_utilization_pct": gpu.gpu_utilization_pct,
            "temperature_c": gpu.temperature_c,
            "power_draw_w": gpu.power_draw_w,
            "cuda_available": gpu.cuda_available,
        },
    }


@app.get("/gpu")
async def gpu_stats():
    gpu = get_gpu_stats()
    return {
        "name": gpu.name,
        "vram_used_mb": gpu.vram_used_mb,
        "vram_free_mb": gpu.vram_free_mb,
        "vram_total_mb": gpu.vram_total_mb,
        "vram_used_pct": gpu.vram_used_pct,
        "gpu_utilization_pct": gpu.gpu_utilization_pct,
        "temperature_c": gpu.temperature_c,
        "power_draw_w": gpu.power_draw_w,
        "cuda_available": gpu.cuda_available,
    }


@app.get("/queue/status")
async def queue_status():
    gpu = get_gpu_stats()
    total_in_system = scheduler.active + scheduler.waiting
    capacity = total_in_system / MAX_CONCURRENT
    return {
        "active_inference": scheduler.active,
        "waiting": scheduler.waiting,
        "max_concurrent": MAX_CONCURRENT,
        "queue_capacity_pct": round(capacity * 100, 1),
        "backpressure_state": scheduler.current_state(),
        "scheduler": "SJF",
        "max_seen_depth": scheduler.max_seen_depth,
        "vram_used_pct": gpu.vram_used_pct,
        "soft_limit_pct": SOFT_LIMIT_PCT * 100,
        "hard_limit_pct": HARD_LIMIT_PCT * 100,
        "vram_oom_threshold_pct": VRAM_OOM_THRESHOLD_PCT,
    }


@app.get("/models")
async def models():
    return {"models": await list_models()}


@app.post("/simulate/vram")
async def simulate_vram(req: VRAMSimRequest):
    memory_gb = req.params_billions * (req.bits / 8) * (1 + req.overhead)
    memory_mb = memory_gb * 1024
    gpu = get_gpu_stats()
    fits_total = memory_mb <= gpu.vram_total_mb
    fits_free = memory_mb <= gpu.vram_free_mb
    quant_label = {4: "Q4/INT4", 8: "INT8/FP8", 16: "FP16/BF16"}.get(req.bits, f"{req.bits}-bit")
    return {
        "params_billions": req.params_billions,
        "bits": req.bits,
        "quant_label": quant_label,
        "overhead_pct": round(req.overhead * 100, 1),
        "estimated_vram_gb": round(memory_gb, 2),
        "estimated_vram_mb": round(memory_mb, 1),
        "gpu_total_mb": gpu.vram_total_mb,
        "gpu_free_mb": gpu.vram_free_mb,
        "fits_in_gpu": fits_total,
        "fits_in_free_vram": fits_free,
        "recommendation": (
            "Fits with headroom" if fits_free
            else "Fits total VRAM but may conflict with running models" if fits_total
            else "Exceeds GPU VRAM — use smaller quant or model"
        ),
        "formula": f"{req.params_billions}B × ({req.bits}/8) × (1 + {req.overhead}) = {memory_gb:.2f} GB",
    }


# ─── Routes: inference ──────────────────────────────────────────────────────
@app.post("/generate")
async def generate_endpoint(req: ChatRequest):
    vram_start = get_gpu_stats().vram_used_mb
    depth_at_arrival = scheduler.active + scheduler.waiting
    ticket = await scheduler.acquire(estimate_job_tokens(req.prompt, req.max_tokens))
    error: Optional[str] = None
    result = None
    try:
        result = await generate(
            prompt=req.prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return {
            "model": result.model,
            "response": result.response,
            "metrics": {
                "ttft_ms": result.ttft_ms,
                "tpot_ms": result.tpot_ms,
                "total_time_s": result.total_time_s,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "tokens_per_second": result.tokens_per_second,
                "queue_wait_ms": round(ticket.queue_wait_ms, 1),
                "scheduler_state_at_acquire": ticket.state_at_acquire,
            },
        }
    except Exception as e:
        error = str(e)
        raise HTTPException(500, error)
    finally:
        scheduler.release()
        await _log_run(
            endpoint="generate",
            model=req.model,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            ttft_ms=result.ttft_ms if result else None,
            tpot_ms=result.tpot_ms if result else None,
            total_time_s=result.total_time_s if result else None,
            tokens_per_second=result.tokens_per_second if result else None,
            input_tokens=result.input_tokens if result else None,
            output_tokens=result.output_tokens if result else None,
            vram_start_mb=vram_start,
            queue_wait_ms=ticket.queue_wait_ms,
            backpressure_state=ticket.state_at_acquire,
            queue_depth_start=depth_at_arrival,
            error=error,
        )


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """SSE streaming; logs final stats to SQLite when done."""
    vram_start = get_gpu_stats().vram_used_mb
    depth_at_arrival = scheduler.active + scheduler.waiting
    ticket = await scheduler.acquire(estimate_job_tokens(req.prompt, req.max_tokens))

    async def event_generator():
        final_stats: dict = {}
        err: Optional[str] = None
        try:
            async for chunk in stream_generate(
                prompt=req.prompt,
                model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system=req.system,
            ):
                # Inject queue-wait + scheduler-state into the first emitted chunk
                if chunk.get("ttft_ms") is not None:
                    chunk["queue_wait_ms"] = round(ticket.queue_wait_ms, 1)
                    chunk["scheduler_state_at_acquire"] = ticket.state_at_acquire
                yield f"data: {json.dumps(chunk)}\n\n"
                if chunk["done"]:
                    final_stats = chunk.get("stats") or {}
                    break
        except Exception as e:
            err = str(e)
            yield f"data: {json.dumps({'error': err, 'done': True})}\n\n"
        finally:
            scheduler.release()
            await _log_run(
                endpoint="chat",
                model=req.model,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                ttft_ms=final_stats.get("ttft_ms"),
                tpot_ms=final_stats.get("tpot_ms"),
                total_time_s=final_stats.get("total_time_s"),
                tokens_per_second=final_stats.get("tokens_per_second"),
                input_tokens=final_stats.get("input_tokens"),
                output_tokens=final_stats.get("output_tokens"),
                vram_start_mb=vram_start,
                queue_wait_ms=ticket.queue_wait_ms,
                backpressure_state=ticket.state_at_acquire,
                queue_depth_start=depth_at_arrival,
                error=err,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Routes: benchmark ──────────────────────────────────────────────────────
@app.post("/benchmark")
async def run_benchmark(req: BenchmarkRequest, bg: BackgroundTasks):
    if _benchmark_state["running"]:
        raise HTTPException(409, "A benchmark is already running.")
    _benchmark_state["running"] = True
    _benchmark_state["progress"] = []
    _benchmark_state["result"] = None

    def on_progress(msg: str):
        _benchmark_state["progress"].append(msg)

    async def _run():
        try:
            result = await benchmark_model(
                model=req.model,
                prompt_type=req.prompt_type,
                runs=req.runs,
                on_progress=on_progress,
            )
            _benchmark_state["result"] = asdict(result)
            # Log each individual run from the benchmark for the research DB
            for raw in result.raw_results:
                await db.log_inference_run(
                    ts=time.time(),
                    endpoint="benchmark",
                    model=req.model,
                    prompt_type=req.prompt_type,
                    prompt_chars=len(BENCHMARK_PROMPTS.get(req.prompt_type, "")),
                    max_tokens=256,
                    temperature=0.1,
                    ttft_ms=raw["ttft_ms"],
                    tpot_ms=raw["tpot_ms"],
                    total_time_s=None,
                    tokens_per_second=raw["tps"],
                    input_tokens=None,
                    output_tokens=raw["output_tokens"],
                    vram_used_mb_start=None,
                    vram_used_mb_end=get_gpu_stats().vram_used_mb,
                    gpu_util_pct_end=get_gpu_stats().gpu_utilization_pct,
                    queue_depth_start=0,
                    queue_wait_ms=0,
                    backpressure_state="NORMAL",
                    error=None,
                )
        except Exception as e:
            _benchmark_state["result"] = {"error": str(e)}
        finally:
            _benchmark_state["running"] = False

    bg.add_task(_run)
    return {"started": True, "model": req.model, "runs": req.runs}


@app.get("/benchmark/status")
async def benchmark_status():
    return {
        "running": _benchmark_state["running"],
        "progress": _benchmark_state["progress"],
        "result": _benchmark_state["result"],
    }


@app.post("/compare")
async def compare_endpoint(req: CompareRequest, bg: BackgroundTasks):
    if _benchmark_state["running"]:
        raise HTTPException(409, "A benchmark is already running.")
    _benchmark_state["running"] = True
    _benchmark_state["progress"] = []
    _benchmark_state["result"] = None

    def on_progress(msg: str):
        _benchmark_state["progress"].append(msg)

    async def _run():
        try:
            report = await compare_models(
                models=req.models,
                prompt_type=req.prompt_type,
                runs=req.runs,
                on_progress=on_progress,
            )
            _benchmark_state["result"] = {
                "models": report.models,
                "fastest_model": report.fastest_model,
                "lowest_ttft_model": report.lowest_ttft_model,
                "summary": report.summary,
                "results": [
                    {
                        "model": r.model,
                        "avg_ttft_ms": r.avg_ttft_ms,
                        "avg_tpot_ms": r.avg_tpot_ms,
                        "avg_tps": r.avg_tps,
                        "min_tps": r.min_tps,
                        "max_tps": r.max_tps,
                    }
                    for r in report.results
                ],
            }
        except Exception as e:
            _benchmark_state["result"] = {"error": str(e)}
        finally:
            _benchmark_state["running"] = False

    bg.add_task(_run)
    return {"started": True, "models": req.models}


# ─── Routes: research-log / history ─────────────────────────────────────────
@app.get("/history/runs")
async def history_runs(
    model: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Recent inference runs (most-recent first)."""
    rows = await db.fetch_recent_runs(model, limit)
    return {"count": len(rows), "runs": rows}


@app.get("/history/aggregate")
async def history_aggregate():
    """Per-model averages across the entire research log."""
    return {"aggregates": await db.fetch_aggregate_by_model()}


@app.get("/history/timeline")
async def history_timeline(
    model: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=10, le=2000),
):
    """Chronological (ts, tps, ttft, vram) for charting."""
    rows = await db.fetch_timeline(model, limit)
    return {"count": len(rows), "points": rows}


@app.get("/history/loadtests")
async def history_loadtests(limit: int = Query(default=20, ge=1, le=200)):
    return {"loadtests": await db.fetch_recent_load_tests(limit)}


# ─── Routes: load test (demonstrates backpressure) ──────────────────────────
@app.post("/loadtest")
async def loadtest(req: LoadTestRequest):
    """
    Fire `concurrency` parallel /generate calls against the local API and
    record how the SJF queue and backpressure react. Returns a timeline.

    This is the empirical companion to the queueing-theory section: you can
    watch state move NORMAL → THROTTLING → SHEDDING in real time, and see
    the resulting latency distribution.
    """
    ts_start = time.time()
    base_url = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"
    timeline: list[dict] = []
    stop_sampling = asyncio.Event()

    async def _sampler():
        """Sample queue + VRAM every 250ms during the test."""
        while not stop_sampling.is_set():
            gpu = get_gpu_stats()
            timeline.append({
                "t": round(time.time() - ts_start, 3),
                "active": scheduler.active,
                "waiting": scheduler.waiting,
                "queue_pct": round((scheduler.active + scheduler.waiting) / MAX_CONCURRENT * 100, 1),
                "state": scheduler.current_state(),
                "vram_pct": gpu.vram_used_pct,
                "gpu_util_pct": gpu.gpu_utilization_pct,
            })
            try:
                await asyncio.wait_for(stop_sampling.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

    async def _fire_one(client: httpx.AsyncClient) -> dict:
        t0 = time.perf_counter()
        payload = {
            "prompt": req.prompt,
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": 0.2,
        }
        try:
            r = await client.post(f"{base_url}/generate", json=payload, timeout=180.0)
            latency = time.perf_counter() - t0
            if r.status_code == 429:
                return {"status": 429, "latency_s": latency}
            if r.status_code >= 400:
                return {"status": r.status_code, "latency_s": latency, "error": r.text[:200]}
            return {"status": 200, "latency_s": latency, "data": r.json()}
        except Exception as e:
            return {"status": "exc", "latency_s": time.perf_counter() - t0, "error": str(e)}

    sampler_task = asyncio.create_task(_sampler())
    try:
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*(_fire_one(client) for _ in range(req.concurrency)))
    finally:
        stop_sampling.set()
        await sampler_task

    ts_end = time.time()
    successful = [r for r in results if r["status"] == 200]
    rejected_429 = [r for r in results if r["status"] == 429]
    errored = [r for r in results if r["status"] not in (200, 429)]
    latencies = sorted(r["latency_s"] for r in successful)

    avg_lat = round(statistics.mean(latencies), 3) if latencies else None
    p50 = round(latencies[len(latencies) // 2], 3) if latencies else None
    p95 = round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 3) if latencies else None
    max_depth = max((s["active"] + s["waiting"] for s in timeline), default=0)
    final_state = timeline[-1]["state"] if timeline else "NORMAL"

    await db.log_load_test(
        ts_start=ts_start,
        ts_end=ts_end,
        model=req.model,
        concurrency=req.concurrency,
        total_requests=len(results),
        successful=len(successful),
        rejected_429=len(rejected_429),
        errored=len(errored),
        avg_latency_s=avg_lat,
        p50_latency_s=p50,
        p95_latency_s=p95,
        max_queue_depth=max_depth,
        final_state=final_state,
        timeline_json=json.dumps(timeline),
    )

    return {
        "model": req.model,
        "concurrency": req.concurrency,
        "duration_s": round(ts_end - ts_start, 2),
        "successful": len(successful),
        "rejected_429": len(rejected_429),
        "errored": len(errored),
        "latency": {
            "avg_s": avg_lat,
            "p50_s": p50,
            "p95_s": p95,
            "min_s": round(min(latencies), 3) if latencies else None,
            "max_s": round(max(latencies), 3) if latencies else None,
        },
        "max_queue_depth_seen": max_depth,
        "final_state": final_state,
        "timeline": timeline,
    }


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
