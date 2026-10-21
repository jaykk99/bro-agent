"""Pull-based GPU training job queue — SQLite-backed, inside bro-agent.

The bro app queues fine-tune jobs; a Colab/Kaggle worker
(notebooks/train-worker.ipynb) claims them over HTTP, trains with Unsloth
QLoRA, uploads the LoRA adapter back to bro-agent, and wipes its disk.
Same pull system as the Forge Asset Pipeline, but fully inside bro-agent:
no Supabase, no Error Inbox, no third-party host.

Auth: every HTTP route goes through the app's access-code gate in main.py.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid

from .memory import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(DB_PATH)), "adapters")


def _connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute(
                "CREATE TABLE IF NOT EXISTS train_jobs ("
                "id TEXT PRIMARY KEY, model_repo TEXT NOT NULL, dataset TEXT, "
                "lora_rank INTEGER NOT NULL DEFAULT 16 CHECK (lora_rank BETWEEN 1 AND 256), "
                "status TEXT NOT NULL DEFAULT 'queued' "
                "CHECK (status IN ('queued','working','done','failed')), "
                "result_url TEXT, metrics_json TEXT, error TEXT, "
                "created_at REAL NOT NULL, updated_at REAL NOT NULL)"
            )
            # Migration: older DBs lack the attempts column.
            cols = {r["name"] for r in _conn.execute("PRAGMA table_info(train_jobs)").fetchall()}
            if "attempts" not in cols:
                _conn.execute("ALTER TABLE train_jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
            _conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_train_jobs_status "
                "ON train_jobs(status, created_at)"
            )
            _conn.commit()
        return _conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("metrics_json"):
        try:
            d["metrics"] = json.loads(d["metrics_json"])
        except Exception:
            d["metrics"] = None
    return d


def create_job(model_repo: str, dataset: str = "", lora_rank: int = 16) -> str:
    jid = uuid.uuid4().hex
    now = time.time()
    try:
        lora_rank = int(lora_rank)
    except (TypeError, ValueError):
        lora_rank = 16
    lora_rank = max(1, min(256, lora_rank))
    c = _connect()
    with _lock:
        c.execute(
            "INSERT INTO train_jobs (id, model_repo, dataset, lora_rank, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (jid, model_repo, dataset, lora_rank, now, now),
        )
        c.commit()
    return jid


def get_job(job_id: str) -> dict | None:
    c = _connect()
    row = c.execute("SELECT * FROM train_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict]:
    c = _connect()
    rows = c.execute(
        "SELECT id, model_repo, dataset, lora_rank, status, result_url,"
        " metrics_json, error, created_at, updated_at"
        " FROM train_jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def claim_job() -> dict | None:
    """Atomically claim the oldest queued job. One statement — first claimer wins,
    safe across processes/workers."""
    c = _connect()
    with _lock:
        row = c.execute(
            "UPDATE train_jobs SET status='working', updated_at=? "
            "WHERE id = (SELECT id FROM train_jobs WHERE status='queued'"
            " ORDER BY created_at ASC LIMIT 1) "
            "RETURNING id, model_repo, dataset, lora_rank, status, created_at, updated_at",
            (time.time(),),
        ).fetchone()
        c.commit()
    return dict(row) if row else None


def complete_job(
    job_id: str,
    status: str,
    result_url: str = "",
    metrics: dict | None = None,
    error: str = "",
) -> bool:
    if status not in ("done", "failed"):
        raise ValueError("status must be done or failed")
    c = _connect()
    with _lock:
        cur = c.execute(
            "UPDATE train_jobs SET status=?, result_url=?, metrics_json=?,"
            " error=?, updated_at=? WHERE id=?",
            (
                status,
                result_url or None,
                json.dumps(metrics) if metrics else None,
                (error or "")[:2000] or None,
                time.time(),
                job_id,
            ),
        )
        c.commit()
        return cur.rowcount > 0


def requeue_job(job_id: str) -> bool:
    """Move a 'working' job back to 'queued' (its worker died). Returns False
    if the job is not currently 'working' (finished jobs are never touched)."""
    c = _connect()
    with _lock:
        cur = c.execute(
            "UPDATE train_jobs SET status='queued', updated_at=? WHERE id=? AND status='working'",
            (time.time(), job_id),
        )
        c.commit()
        return cur.rowcount > 0


def retry_job(job_id: str, max_attempts: int = 3) -> bool:
    """Retry a failed job: failed -> queued, attempts+1. Refuses when the job
    already hit max_attempts (something is really wrong — needs Jay)."""
    c = _connect()
    with _lock:
        row = c.execute(
            "SELECT status, attempts FROM train_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if not row or row["status"] != "failed":
            return False
        if (row["attempts"] or 0) >= max_attempts:
            return False
        c.execute(
            "UPDATE train_jobs SET status='queued', attempts=attempts+1,"
            " error=NULL, updated_at=? WHERE id=?",
            (time.time(), job_id),
        )
        c.commit()
        return True


def fail_stuck_jobs(max_working_s: float = 3600) -> list[str]:
    """Re-queue 'working' jobs whose worker vanished without completing.

    The forge dispatcher reaps dead *workers*; this covers jobs stuck in
    'working' with no worker row at all (e.g. claimed via /api/train/claim
    directly, then the notebook died). Attempts are bumped so a poison job
    can't spin forever — retry_job() still caps total attempts.
    """
    c = _connect()
    cutoff = time.time() - max_working_s
    with _lock:
        stuck = c.execute(
            "SELECT id FROM train_jobs WHERE status='working' AND updated_at < ?",
            (cutoff,),
        ).fetchall()
        ids = [r["id"] for r in stuck]
        if ids:
            c.execute(
                "UPDATE train_jobs SET status='queued', attempts=attempts+1,"
                " updated_at=? WHERE status='working' AND updated_at < ?",
                (time.time(), cutoff),
            )
            c.commit()
        return ids


def count_queued_jobs() -> int:
    """Queue depth for the forge dispatcher status snapshot."""
    c = _connect()
    row = c.execute("SELECT COUNT(*) FROM train_jobs WHERE status='queued'").fetchone()
    return int(row[0]) if row else 0


def adapter_file_path(job_id: str, filename: str) -> str:
    """Sanitized on-disk path for an uploaded adapter file (path traversal safe)."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(filename or ""))[:128] or "file"
    d = os.path.join(ADAPTERS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, safe)
