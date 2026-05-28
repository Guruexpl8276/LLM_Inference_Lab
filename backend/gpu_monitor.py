"""
GPU Monitor — NVIDIA GPU metrics via nvidia-ml-py (pynvml API) + nvidia-smi fallback.
Tracks: VRAM used/free/total, GPU utilization %, temperature, power draw.
Works on any CUDA-capable NVIDIA card; returns a stub when no GPU is detected.
"""

import subprocess
import json
from dataclasses import dataclass
from typing import Optional

# nvidia-ml-py provides the pynvml API (it IS pynvml; the old pynvml package is deprecated).
# Fall back to nvidia-smi subprocess if the package isn't installed.
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage,
        NVML_TEMPERATURE_GPU,
    )
    nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


@dataclass
class GPUStats:
    name: str
    vram_used_mb: float
    vram_free_mb: float
    vram_total_mb: float
    vram_used_pct: float
    gpu_utilization_pct: float
    temperature_c: float
    power_draw_w: Optional[float]
    cuda_available: bool


def get_gpu_stats() -> GPUStats:
    """Return current GPU stats, trying pynvml then nvidia-smi."""
    if PYNVML_AVAILABLE:
        return _stats_via_pynvml()
    return _stats_via_smi()


def _stats_via_pynvml() -> GPUStats:
    try:
        handle = nvmlDeviceGetHandleByIndex(0)
        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()

        mem = nvmlDeviceGetMemoryInfo(handle)
        vram_total = mem.total / 1024 / 1024
        vram_used  = mem.used  / 1024 / 1024
        vram_free  = mem.free  / 1024 / 1024

        util = nvmlDeviceGetUtilizationRates(handle)
        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

        try:
            power = nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            power = None

        return GPUStats(
            name=name,
            vram_used_mb=round(vram_used, 1),
            vram_free_mb=round(vram_free, 1),
            vram_total_mb=round(vram_total, 1),
            vram_used_pct=round((vram_used / vram_total) * 100, 1),
            gpu_utilization_pct=float(util.gpu),
            temperature_c=float(temp),
            power_draw_w=round(power, 1) if power else None,
            cuda_available=True,
        )
    except Exception as e:
        return _error_stats(str(e))


def _stats_via_smi() -> GPUStats:
    """Parse nvidia-smi CSV output."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return _error_stats("nvidia-smi failed")

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        name       = parts[0]
        vram_used  = float(parts[1])
        vram_free  = float(parts[2])
        vram_total = float(parts[3])
        gpu_util   = float(parts[4])
        temp       = float(parts[5])
        power_str  = parts[6]
        power      = float(power_str) if power_str != "[N/A]" else None

        return GPUStats(
            name=name,
            vram_used_mb=vram_used,
            vram_free_mb=vram_free,
            vram_total_mb=vram_total,
            vram_used_pct=round((vram_used / vram_total) * 100, 1),
            gpu_utilization_pct=gpu_util,
            temperature_c=temp,
            power_draw_w=power,
            cuda_available=True,
        )
    except Exception as e:
        return _error_stats(str(e))


def _error_stats(reason: str) -> GPUStats:
    return GPUStats(
        name=f"GPU unavailable ({reason})",
        vram_used_mb=0.0,
        vram_free_mb=0.0,
        vram_total_mb=0.0,
        vram_used_pct=0.0,
        gpu_utilization_pct=0.0,
        temperature_c=0.0,
        power_draw_w=None,
        cuda_available=False,
    )
