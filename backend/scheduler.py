"""
SJF (Shortest Job First) scheduler with backpressure.

Replaces FIFO _acquire_inference_slot. Requests with a smaller estimated job size
(max_tokens) are admitted before larger ones when the queue is contested.

Backpressure preserved:
  - VRAM > VRAM_OOM_THRESHOLD_PCT    → HTTP 429 immediately
  - waiters / MAX > HARD_LIMIT_PCT   → HTTP 429 + Retry-After
  - waiters / MAX > SOFT_LIMIT_PCT   → proportional asyncio.sleep delay
  - capacity available               → SJF wins next slot

All thresholds are tunable via env vars (see .env.example).
Wait time, state, and depth are exposed via /queue/status and DB logging.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

from backend.gpu_monitor import get_gpu_stats


# ─── Tunables (override via env vars) ────────────────────────────────────────
MAX_CONCURRENT          = int(os.getenv("MAX_CONCURRENT", "6"))
SOFT_LIMIT_PCT          = float(os.getenv("SOFT_LIMIT_PCT", "0.50"))
HARD_LIMIT_PCT          = float(os.getenv("HARD_LIMIT_PCT", "0.80"))
VRAM_OOM_THRESHOLD_PCT  = float(os.getenv("VRAM_OOM_THRESHOLD_PCT", "90.0"))


@dataclass
class SlotTicket:
    """Returned to a caller that successfully claimed a slot. Pass to release()."""
    granted_at: float
    queue_wait_ms: float
    state_at_acquire: str         # NORMAL | THROTTLING | SHEDDING


class SJFScheduler:
    """
    Priority queue keyed on (estimated_tokens, fifo_tiebreaker).
    A request that would push us past the hard limit is rejected with HTTP 429.
    """

    def __init__(self) -> None:
        self._active: int = 0
        self._waiting: int = 0
        self._max_seen_depth: int = 0
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._counter = itertools.count()  # FIFO tiebreaker for equal priorities
        # One signaling event per waiter — we wake the smallest-job waiter first.
        self._lock = asyncio.Lock()

    # ── Public introspection ────────────────────────────────────────────────
    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def max_seen_depth(self) -> int:
        return self._max_seen_depth

    def current_state(self) -> str:
        gpu = get_gpu_stats()
        capacity = (self._active + self._waiting) / MAX_CONCURRENT
        if capacity >= HARD_LIMIT_PCT or gpu.vram_used_pct >= VRAM_OOM_THRESHOLD_PCT:
            return "SHEDDING"
        if capacity >= SOFT_LIMIT_PCT:
            return "THROTTLING"
        return "NORMAL"

    # ── Acquire / release ───────────────────────────────────────────────────
    async def acquire(self, estimated_tokens: int) -> SlotTicket:
        """
        Claim a slot. Smallest estimated_tokens wins on contention.
        Raises HTTPException(429) when shedding.
        """
        t_arrive = time.perf_counter()
        gpu = get_gpu_stats()
        capacity_now = (self._active + self._waiting) / MAX_CONCURRENT

        # Hard reject zone — load shedding
        if capacity_now >= HARD_LIMIT_PCT or gpu.vram_used_pct >= VRAM_OOM_THRESHOLD_PCT:
            retry_after = max(3, int(capacity_now * 15))
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Inference queue saturated — load shedding active",
                    "active": self._active,
                    "waiting": self._waiting,
                    "max": MAX_CONCURRENT,
                    "queue_pct": round(capacity_now * 100, 1),
                    "vram_pct": gpu.vram_used_pct,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Throttle zone — proportional intake delay
        if capacity_now >= SOFT_LIMIT_PCT:
            delay = (capacity_now - SOFT_LIMIT_PCT) * 0.4   # 0s @ 50% → 120ms @ 80%
            await asyncio.sleep(delay)

        # Fast path: capacity free, no waiters → take slot immediately
        if self._active < MAX_CONCURRENT and self._waiting == 0:
            self._active += 1
            return SlotTicket(
                granted_at=time.perf_counter(),
                queue_wait_ms=(time.perf_counter() - t_arrive) * 1000,
                state_at_acquire="NORMAL" if capacity_now < SOFT_LIMIT_PCT else "THROTTLING",
            )

        # Contended path: enqueue with SJF priority and wait for signal
        my_event = asyncio.Event()
        priority = (max(1, int(estimated_tokens)), next(self._counter))
        await self._queue.put((priority, my_event))
        self._waiting += 1
        self._max_seen_depth = max(self._max_seen_depth, self._active + self._waiting)

        try:
            # Wake-up loop: if there's room and *I* am the smallest waiter, take it.
            while True:
                # Check head of queue — am I next?
                if self._active < MAX_CONCURRENT:
                    # Peek the smallest-priority waiter
                    next_priority, next_event = self._queue._queue[0]
                    if next_event is my_event:
                        # I'm next → pop and take slot
                        self._queue.get_nowait()
                        self._queue.task_done()
                        self._active += 1
                        self._waiting -= 1
                        wait_ms = (time.perf_counter() - t_arrive) * 1000
                        return SlotTicket(
                            granted_at=time.perf_counter(),
                            queue_wait_ms=wait_ms,
                            state_at_acquire=self.current_state(),
                        )
                # Not yet — wait until someone releases
                await my_event.wait()
                my_event.clear()
        except asyncio.CancelledError:
            # Caller went away (client disconnect) — remove our entry
            self._waiting = max(0, self._waiting - 1)
            try:
                # Best-effort drain — remove our event from queue
                items = []
                while not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item[1] is not my_event:
                        items.append(item)
                    self._queue.task_done()
                for item in items:
                    await self._queue.put(item)
            except Exception:
                pass
            raise

    def release(self) -> None:
        """Free a slot and wake the smallest-job waiter."""
        self._active = max(0, self._active - 1)
        if not self._queue.empty():
            # Wake the head waiter (smallest priority)
            try:
                _, next_event = self._queue._queue[0]
                next_event.set()
            except Exception:
                pass


# ─── Singleton ───────────────────────────────────────────────────────────────
scheduler = SJFScheduler()


def estimate_job_tokens(prompt: str, max_tokens: int) -> int:
    """
    Rough job-cost proxy = max_tokens (output dominates total wallclock).
    Adds a small contribution for the prompt to differentiate edge cases.
    """
    return max_tokens + max(1, len(prompt) // 16)
