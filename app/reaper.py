"""Automatic stuck-job reaper: background loop for the train queue + forge dispatcher.

The manual endpoints (POST /api/train/reap-stuck, POST /api/forge/dispatch/reap)
only run when someone remembers to call them. Colab/Kaggle workers die
without notice (timeouts, idle disconnects), and on Render's free tier the
app sleeps between polls — a dead worker's job could sit 'working' for days
before anyone hit reap, while its GPU slot was already lost. This loop runs
both reaps on a timer inside the app itself, so stuck jobs get re-queued
(or retired at the attempts cap) with no operator action.

Started from the FastAPI lifespan in app/main.py; stops on shutdown.
Disabled with BRO_REAP_INTERVAL_S=0. The interval clamps to >= 60s so a
typo can't hammer the DB. Reaping is idempotent (both halves only touch
jobs/workers in the stuck state), so a restart, a manual reap, and this
loop can all overlap safely — the second caller simply finds nothing to
do. One half failing (DB error) is logged and recorded in status() but
never kills the loop or skips the other half.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 600
MIN_INTERVAL_S = 60

# One entry per run_reap_once() call, even when both halves find nothing.
_state: dict = {
    "runs": 0,
    "last_run_at": None,  # epoch seconds of the last reap pass
    "last_result": None,  # {"requeued": [...], "failed": [...], "reaped_workers": n}
    "last_error": None,  # first error string from the last pass, else None
}

_task: asyncio.Task | None = None
_task_interval_s: float | None = None


def reap_interval_s() -> int:
    """Reap interval from BRO_REAP_INTERVAL_S. Default 600s; 0 disables;
    1-59s clamps up to 60s; garbage falls back to the default."""
    try:
        raw = int(os.environ.get("BRO_REAP_INTERVAL_S", str(DEFAULT_INTERVAL_S)))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S
    if raw <= 0:
        return 0
    return max(MIN_INTERVAL_S, raw)


def run_reap_once(max_working_s: float = 3600) -> dict:
    """One reap pass: stuck train jobs, then dead forge workers.

    Returns {"requeued": [job ids put back to 'queued'],
             "failed": [job ids failed at the attempts cap],
             "reaped_workers": <workers marked dead>}. A half that raises
    is logged, recorded in status(), and never blocks the other half —
    a DB blip on the train side must not leave dead workers unreaped.
    """
    from . import dispatch as forge_dispatch
    from . import train as train_jobs

    requeued: list[str] = []
    failed: list[str] = []
    reaped_workers = 0
    first_error: str | None = None

    try:
        stuck = train_jobs.fail_stuck_jobs(max_working_s)
        requeued.extend(stuck.get("requeued", []))
        failed.extend(stuck.get("failed", []))
    except Exception as e:  # keep the forge half running
        first_error = f"train reap failed: {e}"
        logger.exception("auto-reaper: train half failed")

    try:
        forge = forge_dispatch.reap_dead_workers()
        reaped_workers = int(forge.get("reaped", 0))
        requeued.extend(forge.get("requeued", []))
        failed.extend(forge.get("failed", []))
    except Exception as e:
        if first_error is None:
            first_error = f"forge reap failed: {e}"
        logger.exception("auto-reaper: forge half failed")

    result = {
        "requeued": list(dict.fromkeys(requeued)),
        "failed": list(dict.fromkeys(failed)),
        "reaped_workers": reaped_workers,
    }
    _state["runs"] += 1
    _state["last_run_at"] = time.time()
    _state["last_result"] = result
    _state["last_error"] = first_error
    if requeued or failed or reaped_workers:
        logger.info(
            "auto-reaper: requeued=%d failed=%d reaped_workers=%d",
            len(result["requeued"]),
            len(result["failed"]),
            reaped_workers,
        )
    return result


def status() -> dict:
    """Reaper state for ops: enabled/interval, whether the loop is live,
    how many passes ran, and what the last pass did (or failed with)."""
    return {
        "enabled": reap_interval_s() > 0,
        "interval_s": reap_interval_s(),
        "running": _task is not None and not _task.done(),
        "runs": _state["runs"],
        "last_run_at": _state["last_run_at"],
        "last_result": _state["last_result"],
        "last_error": _state["last_error"],
    }


async def _loop(interval_s: float) -> None:
    # Reap once on start: a restart (or a wake from Render sleep) should
    # clear the backlog immediately, not after the first full interval.
    while True:
        await asyncio.to_thread(run_reap_once)
        await asyncio.sleep(interval_s)


def start(interval_s: float | None = None) -> bool:
    """Start the background reaper loop. Idempotent: a second call while
    the loop is live returns False. Returns False when disabled
    (BRO_REAP_INTERVAL_S=0) or already running. Must be called from a
    running event loop (the FastAPI lifespan). Pass interval_s explicitly
    to override the env interval (tests, one-off runs); the env path
    clamps to >= 60s, an explicit value is used as given."""
    global _task, _task_interval_s
    if _task is not None and not _task.done():
        return False
    interval = reap_interval_s() if interval_s is None else float(interval_s)
    if interval <= 0:
        return False
    loop = asyncio.get_running_loop()
    _task_interval_s = interval
    _task = loop.create_task(_loop(interval))
    logger.info("auto-reaper started, interval=%ss", interval)
    return True


async def stop() -> None:
    """Stop the background loop (lifespan shutdown). No-op when not running."""
    global _task, _task_interval_s
    task = _task
    _task = None
    _task_interval_s = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("auto-reaper stopped")
