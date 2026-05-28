"""
Persistent research log — async SQLite.
Logs every inference run and load test for longitudinal analysis.

Schema is intentionally denormalized for fast dashboard reads — one row per run.
Uses aiosqlite to keep FastAPI's event loop unblocked.

Default DB location: ./inference_lab.db (project root).
Override with INFERENCE_DB_PATH env var if you want to store it elsewhere
(e.g. on a separate drive — see .env.example).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncIterator

import aiosqlite


_DEFAULT_DB = Path(__file__).resolve().parent.parent / "inference_lab.db"
DB_PATH = Path(os.getenv("INFERENCE_DB_PATH", str(_DEFAULT_DB)))


_INIT_SQL = """
CREATE TABLE IF NOT EXISTS inference_runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    REAL    NOT NULL,                -- unix epoch seconds
    endpoint              TEXT    NOT NULL,                -- 'chat' | 'generate' | 'benchmark'
    model                 TEXT    NOT NULL,
    prompt_type           TEXT,                            -- short|medium|long|adhoc
    prompt_chars          INTEGER,
    max_tokens            INTEGER,
    temperature           REAL,
    ttft_ms               REAL,
    tpot_ms               REAL,
    total_time_s          REAL,
    tokens_per_second     REAL,
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    vram_used_mb_start    REAL,
    vram_used_mb_end      REAL,
    gpu_util_pct_end      REAL,
    queue_depth_start     INTEGER,
    queue_wait_ms         REAL,                            -- time spent waiting for slot
    backpressure_state    TEXT,                            -- NORMAL|THROTTLING|SHEDDING
    error                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_model ON inference_runs(model);
CREATE INDEX IF NOT EXISTS idx_runs_ts    ON inference_runs(ts);

CREATE TABLE IF NOT EXISTS load_tests (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start              REAL    NOT NULL,
    ts_end                REAL    NOT NULL,
    model                 TEXT    NOT NULL,
    concurrency           INTEGER NOT NULL,
    total_requests        INTEGER NOT NULL,
    successful            INTEGER NOT NULL,
    rejected_429          INTEGER NOT NULL,
    errored               INTEGER NOT NULL,
    avg_latency_s         REAL,
    p50_latency_s         REAL,
    p95_latency_s         REAL,
    max_queue_depth       INTEGER,
    final_state           TEXT,
    timeline_json         TEXT                              -- JSON: [{t, queue, vram, state}, ...]
);
"""


async def init_db() -> None:
    """Create schema if missing. Idempotent."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_INIT_SQL)
        await db.commit()


@asynccontextmanager
async def connect() -> AsyncIterator[aiosqlite.Connection]:
    """Yield a connection with row_factory set so rows behave like dicts."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def log_inference_run(
    *,
    ts: float,
    endpoint: str,
    model: str,
    prompt_type: Optional[str],
    prompt_chars: Optional[int],
    max_tokens: Optional[int],
    temperature: Optional[float],
    ttft_ms: Optional[float],
    tpot_ms: Optional[float],
    total_time_s: Optional[float],
    tokens_per_second: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    vram_used_mb_start: Optional[float],
    vram_used_mb_end: Optional[float],
    gpu_util_pct_end: Optional[float],
    queue_depth_start: Optional[int],
    queue_wait_ms: Optional[float],
    backpressure_state: Optional[str],
    error: Optional[str] = None,
) -> None:
    """Insert one inference run. Swallows DB errors — logging must never break inference."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO inference_runs (
                    ts, endpoint, model, prompt_type, prompt_chars, max_tokens, temperature,
                    ttft_ms, tpot_ms, total_time_s, tokens_per_second,
                    input_tokens, output_tokens,
                    vram_used_mb_start, vram_used_mb_end, gpu_util_pct_end,
                    queue_depth_start, queue_wait_ms, backpressure_state, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts, endpoint, model, prompt_type, prompt_chars, max_tokens, temperature,
                    ttft_ms, tpot_ms, total_time_s, tokens_per_second,
                    input_tokens, output_tokens,
                    vram_used_mb_start, vram_used_mb_end, gpu_util_pct_end,
                    queue_depth_start, queue_wait_ms, backpressure_state, error,
                ),
            )
            await db.commit()
    except Exception:
        pass


async def log_load_test(
    *,
    ts_start: float,
    ts_end: float,
    model: str,
    concurrency: int,
    total_requests: int,
    successful: int,
    rejected_429: int,
    errored: int,
    avg_latency_s: Optional[float],
    p50_latency_s: Optional[float],
    p95_latency_s: Optional[float],
    max_queue_depth: int,
    final_state: str,
    timeline_json: str,
) -> None:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO load_tests (
                    ts_start, ts_end, model, concurrency, total_requests,
                    successful, rejected_429, errored,
                    avg_latency_s, p50_latency_s, p95_latency_s,
                    max_queue_depth, final_state, timeline_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts_start, ts_end, model, concurrency, total_requests,
                    successful, rejected_429, errored,
                    avg_latency_s, p50_latency_s, p95_latency_s,
                    max_queue_depth, final_state, timeline_json,
                ),
            )
            await db.commit()
    except Exception:
        pass


async def fetch_recent_runs(model: Optional[str], limit: int) -> list[dict]:
    sql = "SELECT * FROM inference_runs"
    params: tuple = ()
    if model:
        sql += " WHERE model = ?"
        params = (model,)
    sql += " ORDER BY ts DESC LIMIT ?"
    params = params + (limit,)
    async with connect() as db:
        cur = await db.execute(sql, params)
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def fetch_aggregate_by_model() -> list[dict]:
    """Per-model averages across all logged runs (excludes errored runs)."""
    sql = """
    SELECT
        model,
        COUNT(*)                            AS runs,
        ROUND(AVG(ttft_ms), 1)              AS avg_ttft_ms,
        ROUND(AVG(tpot_ms), 2)              AS avg_tpot_ms,
        ROUND(AVG(tokens_per_second), 2)    AS avg_tps,
        ROUND(MIN(tokens_per_second), 2)    AS min_tps,
        ROUND(MAX(tokens_per_second), 2)    AS max_tps,
        ROUND(AVG(vram_used_mb_end), 1)     AS avg_vram_mb,
        ROUND(AVG(queue_wait_ms), 1)        AS avg_queue_wait_ms
    FROM inference_runs
    WHERE error IS NULL AND tokens_per_second IS NOT NULL
    GROUP BY model
    ORDER BY avg_tps DESC
    """
    async with connect() as db:
        cur = await db.execute(sql)
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def fetch_timeline(model: Optional[str], limit: int) -> list[dict]:
    """Recent runs as a (ts, tps, ttft_ms, vram) timeline for charts."""
    sql = """
    SELECT ts, model, tokens_per_second AS tps, ttft_ms, vram_used_mb_end AS vram
    FROM inference_runs
    WHERE error IS NULL AND tokens_per_second IS NOT NULL
    """
    params: tuple = ()
    if model:
        sql += " AND model = ?"
        params = (model,)
    sql += " ORDER BY ts DESC LIMIT ?"
    params = params + (limit,)
    async with connect() as db:
        cur = await db.execute(sql, params)
        rows = await cur.fetchall()
        return list(reversed([dict(r) for r in rows]))  # chronological


async def fetch_recent_load_tests(limit: int) -> list[dict]:
    async with connect() as db:
        cur = await db.execute(
            "SELECT id, ts_start, ts_end, model, concurrency, total_requests, "
            "successful, rejected_429, errored, avg_latency_s, p50_latency_s, "
            "p95_latency_s, max_queue_depth, final_state "
            "FROM load_tests ORDER BY ts_start DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]
