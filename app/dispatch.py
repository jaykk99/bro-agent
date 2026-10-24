"""Forge worker dispatcher — registry, heartbeats, and dead-worker reaper.

The Forge Asset Pipeline's job queue lives on Supabase
(`forge_asset_jobs`) and workers claim jobs with an atomic pull (first
claimer wins), which survives restarts. This module is the registry and
monitoring layer on top of that pull model:

- Workers register (platform: colab/kaggle) and heartbeat every ~30s with
  the job they are currently processing (if any).
- The dispatcher tracks which workers are alive per platform.
- A reaper detects dead workers (stale heartbeat) and re-queues their
  in-flight jobs back to 'queued' on Supabase, so no job is lost when a
  Colab/Kaggle runtime dies mid-generation.

Job claiming stays pull-based and atomic — the dispatcher never pushes
jobs, it keeps the queue healthy. SQLite-backed, same DB as train.py.

This module was missing from git at HEAD even though app/main.py imports
it, so a fresh clone could not start the app at all. It is written here
from main.py's call sites, not from any uncommitted scratch copy.

Auth: every HTTP route goes through the app's access-code gate in main.py.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid

from .memory import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# A worker is 'alive' if it heartbeated within this window.
ALIVE_SECS = 90
# A worker is reaped (dead + job re-queued) after this long without heartbeat.
REAP_SECS = 180

_KNOWN_PLATFORMS = ("colab", "kaggle")


def _connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute(
                "CREATE TABLE IF NOT EXISTS forge_workers ("
                "worker_id TEXT PRIMARY KEY, platform TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'alive', "
                "last_heartbeat REAL NOT NULL, current_job_id TEXT, "
                "jobs_done INTEGER NOT NULL DEFAULT 0, "
                "created_at REAL NOT NULL)"
            )
            _conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forge_workers_heartbeat "
                "ON forge_workers(last_heartbeat)"
            )
            _conn.commit()
        return _conn


def _normalize_platform(platform: str) -> str:
    p = (platform or "").strip().lower()
    return p if p in _KNOWN_PLATFORMS else "unknown"


def register_worker(platform: str) -> str:
    """Register a new worker; returns its worker_id."""
    worker_id = uuid.uuid4().hex[:12]
    c = _connect()
    now = time.time()
    with _lock:
        c.execute(
            "INSERT INTO forge_workers (worker_id, platform, status, "
            "last_heartbeat, current_job_id, jobs_done, created_at) "
            "VALUES (?, ?, 'alive', ?, NULL, 0, ?)",
            (worker_id, _normalize_platform(platform), now, now),
        )
        c.commit()
    return worker_id


def heartbeat(worker_id: str, current_job_id: str = "") -> bool:
    """Record a heartbeat. Empty current_job_id clears the in-flight job.

    Returns False for an unknown worker_id so the route can 404.
    """
    c = _connect()
    with _lock:
        cur = c.execute(
            "UPDATE forge_workers SET last_heartbeat=?, status='alive', "
            "current_job_id=? WHERE worker_id=?",
            (time.time(), current_job_id or None, worker_id),
        )
        c.commit()
        return cur.rowcount > 0


def job_finished(worker_id: str) -> None:
    """Mark the worker's current job done: count it, clear the slot."""
    c = _connect()
    with _lock:
        c.execute(
            "UPDATE forge_workers SET jobs_done = jobs_done + 1, "
            "current_job_id=NULL WHERE worker_id=?",
            (worker_id,),
        )
        c.commit()


def list_workers() -> list[dict]:
    """All registered workers, each with a computed 'alive' flag."""
    c = _connect()
    now = time.time()
    rows = c.execute(
        "SELECT worker_id, platform, status, last_heartbeat, "
        "current_job_id, jobs_done, created_at FROM forge_workers "
        "ORDER BY last_heartbeat DESC"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["alive"] = bool(now - d["last_heartbeat"] <= ALIVE_SECS)
        out.append(d)
    return out


def _queue_depth() -> int:
    """Jobs currently queued in the local train_jobs table, if it exists."""
    c = _connect()
    try:
        row = c.execute(
            "SELECT COUNT(*) FROM train_jobs WHERE status='queued'"
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def status_snapshot() -> dict:
    """Dispatcher status for the ops dashboard: workers + queue depth."""
    workers = list_workers()
    alive_by_platform: dict[str, int] = {}
    for w in workers:
        if w["alive"]:
            alive_by_platform[w["platform"]] = alive_by_platform.get(w["platform"], 0) + 1
    queued = _queue_depth()
    return {
        "workers": workers,
        "alive_by_platform": alive_by_platform,
        "queue_depth": queued,
        "queue": {"queued": queued},
    }


def _supabase_config() -> tuple[str, str] | None:
    """(base_url, key) or None when Supabase is not configured."""
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    return url, key


def _requeue_supabase_job(job_id: str, cfg: tuple[str, str]) -> bool:
    """Best-effort: set one Supabase forge_asset_jobs row back to 'queued'.

    Only touches rows still marked 'working' so a job claimed by a live
    worker after the reaper ran is never clobbered. Never raises.
    """
    url, key = cfg
    req = urllib.request.Request(
        f"{url}/rest/v1/forge_asset_jobs?id=eq.{job_id}&status=eq.working",
        data=json.dumps({"status": "queued"}).encode(),
        method="PATCH",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def reap_dead_workers() -> dict:
    """Mark stale workers dead; re-queue their stuck jobs on Supabase.

    A worker counts as dead only after REAP_SECS without a heartbeat, so a
    worker between heartbeats (ALIVE_SECS window) is never reaped. Returns
    a report that says honestly whether Supabase re-queueing ran.
    """
    c = _connect()
    now = time.time()
    cfg = _supabase_config()
    with _lock:
        stale = c.execute(
            "SELECT worker_id, current_job_id FROM forge_workers "
            "WHERE status='alive' AND last_heartbeat < ?",
            (now - REAP_SECS,),
        ).fetchall()
        reaped = []
        requeued = []
        for row in stale:
            job_id = row["current_job_id"]
            ok = False
            if job_id and cfg is not None:
                ok = _requeue_supabase_job(job_id, cfg)
                if ok:
                    requeued.append(job_id)
            c.execute(
                "UPDATE forge_workers SET status='dead', current_job_id=NULL "
                "WHERE worker_id=?",
                (row["worker_id"],),
            )
            reaped.append(
                {"worker_id": row["worker_id"], "job_id": job_id, "requeued": ok}
            )
        c.commit()
    return {
        "reaped_workers": reaped,
        "requeued_jobs": requeued,
        "supabase_configured": cfg is not None,
    }
