"""
Inference Engine — Ollama wrapper with streaming + perf metrics.
Works with any Ollama-served model. Configure via OLLAMA_HOST / DEFAULT_MODEL env vars.
"""

import json
import os
import time
import httpx
from typing import AsyncGenerator, Optional
from dataclasses import dataclass


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:3b")


@dataclass
class InferenceResult:
    model: str
    prompt: str
    response: str
    ttft_ms: float          # Time to First Token (ms)
    tpot_ms: float          # Time Per Output Token (ms)
    total_time_s: float     # Wall-clock total time (s)
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    done: bool = True


async def stream_generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Streaming token generator. Yields dicts:
      {"token": str, "ttft_ms": float|None, "done": bool, "stats": dict|None}
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "99")),  # 99 = offload all layers
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "2048")),
        },
    }
    if system:
        payload["system"] = system

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    token_count = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{OLLAMA_HOST}/api/generate", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                token_text = chunk.get("response", "")
                done = chunk.get("done", False)

                now = time.perf_counter()

                if token_text and t_first_token is None:
                    t_first_token = now
                    ttft_ms = (t_first_token - t_start) * 1000
                else:
                    ttft_ms = None

                if token_text:
                    token_count += 1

                stats = None
                if done:
                    total_s = now - t_start
                    first_token_s = t_first_token or now
                    generation_time = now - first_token_s
                    tpot = (generation_time / max(token_count - 1, 1)) * 1000
                    tps  = token_count / total_s if total_s > 0 else 0

                    stats = {
                        "ttft_ms": round((t_first_token - t_start) * 1000, 1) if t_first_token else 0,
                        "tpot_ms": round(tpot, 2),
                        "total_time_s": round(total_s, 2),
                        "tokens_per_second": round(tps, 2),
                        "output_tokens": chunk.get("eval_count", token_count),
                        "input_tokens": chunk.get("prompt_eval_count", 0),
                    }

                yield {
                    "token": token_text,
                    "ttft_ms": ttft_ms,
                    "done": done,
                    "stats": stats,
                }

                if done:
                    break


async def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> InferenceResult:
    """Non-streaming full generation with precise timing metrics."""
    full_response = ""
    stats = {}
    first_token_ms = None

    async for chunk in stream_generate(prompt, model, max_tokens, temperature):
        full_response += chunk["token"]
        if chunk["ttft_ms"] is not None:
            first_token_ms = chunk["ttft_ms"]
        if chunk["done"] and chunk["stats"]:
            stats = chunk["stats"]

    return InferenceResult(
        model=model,
        prompt=prompt,
        response=full_response,
        ttft_ms=stats.get("ttft_ms", first_token_ms or 0),
        tpot_ms=stats.get("tpot_ms", 0),
        total_time_s=stats.get("total_time_s", 0),
        input_tokens=stats.get("input_tokens", 0),
        output_tokens=stats.get("output_tokens", 0),
        tokens_per_second=stats.get("tokens_per_second", 0),
        done=True,
    )


async def list_models() -> list[dict]:
    """List all locally available Ollama models."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                size_gb = m.get("size", 0) / 1024 / 1024 / 1024
                models.append({
                    "name": m["name"],
                    "size_gb": round(size_gb, 2),
                    "modified_at": m.get("modified_at", ""),
                })
            return models
        except Exception as e:
            return [{"error": str(e)}]


async def check_ollama_health() -> dict:
    """Check if Ollama is running and responding."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            return {"status": "online", "url": OLLAMA_HOST}
        except httpx.ConnectError:
            return {"status": "offline", "url": OLLAMA_HOST, "error": "Cannot connect to Ollama. Is it running?"}
        except Exception as e:
            return {"status": "error", "url": OLLAMA_HOST, "error": str(e)}
