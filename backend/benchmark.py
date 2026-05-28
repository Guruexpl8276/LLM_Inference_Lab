"""
Benchmark Engine — Multi-run, multi-model performance evaluation.
Measures: TTFT, TPOT, TPS (tokens/sec) across all installed models.
Designed for systematic research: compare quantization, model size effects, scheduling impact.
"""

import asyncio
import time
import statistics
from typing import Optional
from dataclasses import dataclass, field

from backend.inference import generate, list_models


# ─── Standard benchmark prompts (varied complexity) ──────────────────────────
BENCHMARK_PROMPTS = {
    "short": "What is 2 + 2? Answer briefly.",
    "medium": "Explain what VRAM is and why it matters for AI inference in 3 sentences.",
    "long": (
        "Write a detailed explanation of how transformer attention mechanisms work, "
        "including the role of query, key, and value matrices in self-attention, "
        "and how this enables parallel processing compared to RNNs."
    ),
}


@dataclass
class ModelBenchmarkResult:
    model: str
    prompt_type: str
    avg_ttft_ms: float
    avg_tpot_ms: float
    avg_tps: float
    min_tps: float
    max_tps: float
    avg_output_tokens: float
    runs: int
    raw_results: list = field(default_factory=list)


@dataclass
class ComparisonReport:
    models: list[str]
    results: list[ModelBenchmarkResult]
    fastest_model: str
    lowest_ttft_model: str
    summary: str


async def benchmark_model(
    model: str,
    prompt_type: str = "medium",
    runs: int = 3,
    max_tokens: int = 256,
    on_progress=None,
) -> ModelBenchmarkResult:
    """Run N inference passes on a model, return aggregated metrics."""
    prompt = BENCHMARK_PROMPTS.get(prompt_type, BENCHMARK_PROMPTS["medium"])
    raw = []

    for i in range(runs):
        if on_progress:
            on_progress(f"Run {i+1}/{runs} on {model}...")
        result = await generate(prompt, model=model, max_tokens=max_tokens, temperature=0.1)
        raw.append({
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "tps": result.tokens_per_second,
            "output_tokens": result.output_tokens,
        })
        # Small pause between runs to let GPU cool
        await asyncio.sleep(0.5)

    return ModelBenchmarkResult(
        model=model,
        prompt_type=prompt_type,
        avg_ttft_ms=round(statistics.mean(r["ttft_ms"] for r in raw), 1),
        avg_tpot_ms=round(statistics.mean(r["tpot_ms"] for r in raw), 2),
        avg_tps=round(statistics.mean(r["tps"] for r in raw), 2),
        min_tps=round(min(r["tps"] for r in raw), 2),
        max_tps=round(max(r["tps"] for r in raw), 2),
        avg_output_tokens=round(statistics.mean(r["output_tokens"] for r in raw), 1),
        runs=runs,
        raw_results=raw,
    )


async def compare_models(
    models: Optional[list[str]] = None,
    prompt_type: str = "medium",
    runs: int = 2,
    on_progress=None,
) -> ComparisonReport:
    """
    Benchmark multiple models side-by-side.
    If models=None, auto-detect installed models (GPU-safe ones only).
    """
    # Small / mid-size model allow list (substring match — any pull matching these is in).
    # These fit comfortably on 4 GB+ GPUs. Add larger models here on bigger cards.
    SAFE_MODELS = {
        "llama3.2:3b", "llama3.2:1b",
        "phi3:mini", "phi3.5:mini",
        "qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5-coder",
        "gemma2:2b", "gemma3:1b",
        "tinyllama", "smollm2",
        "deepseek-r1:1.5b", "deepseek-coder:1.3b",
    }

    if models is None:
        available = await list_models()
        models = [
            m["name"] for m in available
            if not m.get("error") and any(safe in m["name"] for safe in SAFE_MODELS)
        ]
        if not models:
            models = ["llama3.2:3b"]

    results = []
    for model in models:
        if on_progress:
            on_progress(f"Benchmarking {model}...")
        try:
            r = await benchmark_model(model, prompt_type, runs, on_progress=on_progress)
            results.append(r)
        except Exception as e:
            if on_progress:
                on_progress(f"FAILED {model}: {e}")

    if not results:
        return ComparisonReport(
            models=models, results=[], fastest_model="N/A",
            lowest_ttft_model="N/A", summary="No results."
        )

    fastest = max(results, key=lambda r: r.avg_tps)
    lowest_ttft = min(results, key=lambda r: r.avg_ttft_ms)

    summary_lines = [f"Benchmark: {prompt_type} prompt, {runs} runs each"]
    for r in sorted(results, key=lambda x: x.avg_tps, reverse=True):
        summary_lines.append(
            f"  {r.model}: {r.avg_tps:.1f} tok/s | TTFT={r.avg_ttft_ms:.0f}ms | TPOT={r.avg_tpot_ms:.1f}ms"
        )

    return ComparisonReport(
        models=models,
        results=results,
        fastest_model=fastest.model,
        lowest_ttft_model=lowest_ttft.model,
        summary="\n".join(summary_lines),
    )
