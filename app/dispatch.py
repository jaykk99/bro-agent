"""Forge worker dispatcher: Colab/Kaggle worker registry, heartbeats, reaper.

The train-worker notebooks (notebooks/train-worker.ipynb) register here,
send heartbeats while they run, and the reaper marks workers dead when they
stop checking in, re-queueing any train job they left stuck in 'working' so
another worker can claim it.

Auth: the /api/forge routes in app/main.py sit behind the app's access-code
gate. Storage: SQLite forge_workers table in the same DB as threads and
train jobs (see app/memory.py BRO_DB_PATH). Render free-tier disk is
ephemeral (same caveat as the rest of bro-agent).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid

from .memory import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# Workers heartbeat every ~30s; allow generous slack for free-tier sleeps
# and Render's cold-start wakeups.
HEARTBEAT_TIMEOUT_S = 600


def _connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute(
                "CREATE TABLE IF NOT EXISTS forge_workers ("
                "id TEXT PRIMARY KEY, platform TEXT NOT NULL DEFAULT 'other', "
                "last_seen REAL NOT NULL, current_job_id TEXT, "
                "dead INTEGER NOT NULL DEFAULT 0)"
            )
            _conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forge_workers_seen "
                "ON forge_workers(last_seen, dead)"
            )
            _conn.commit()
        return _conn


def register_worker(platform: str = "") -> str:
    """Register a new worker. Returns its worker_id."""
    wid = uuid.uuid4().hex[:16]
    plat = (platform or "").strip().lower()
    if plat not in ("colab", "kaggle"):
        plat = "other"
    c = _connect()
    with _lock:
        c.execute(
            "INSERT INTO forge_workers (id, platform, last_seen, current_job_id, dead)"
            " VALUES (?, ?, ?, '', 0)",
            (wid, plat, time.time()),
        )
        c.commit()
    return wid


def heartbeat(worker_id: str, current_job_id: str = "") -> bool:
    """Worker check-in. Returns False for an unknown worker_id — and for
    a DEAD one. A reaped worker that wakes up (a Colab runtime resumed
    after its job was re-queued to someone else) must re-register for a
    fresh id instead of silently reviving: the old code cleared dead=0
    on any heartbeat, which also restored the worker's stale
    current_job_id — and the next reap then re-queued the NEW holder's
    job, because requeue_job only checks that the job is 'working'. A
    dead id stays dead; the notebook re-registers when its heartbeat
    is refused (see notebooks/train-worker.ipynb).

    A live check-in also keeps the worker's current train job alive:
    the forge heartbeat carries current_job_id, and a training run
    takes 30-120 min while fail_stuck_jobs reaps 'working' jobs
    untouched for 3600s — before this fix only the worker row was
    refreshed here, so a worker checking in ONLY through this endpoint
    (the documented worker check-in) had its job re-queued mid-training
    and a second worker claimed it, wasting the first worker's GPU run
    and adapter upload (its eventual complete was refused as stale).
    The notebook worked around it by ALSO pinging
    /api/train/{id}/heartbeat, but the server has everything it needs
    right here, so this endpoint is now sufficient on its own. The
    touch goes through train.heartbeat_job, which only refreshes a job
    that is still 'working' AND owned by this worker — a stale worker
    heartbeating with a job another worker now holds cannot keep that
    job alive, and a reaped/finished job is never revived. A False
    there is ignored: the worker heartbeat itself already succeeded."""
    from . import train as train_jobs

    c = _connect()
    with _lock:
        row = c.execute(
            "SELECT dead FROM forge_workers WHERE id=?", (worker_id,)
        ).fetchone()
        if not row or row["dead"]:
            return False
        cur = c.execute(
            "UPDATE forge_workers SET last_seen=?, current_job_id=?, dead=0 WHERE id=?",
            (time.time(), (current_job_id or "").strip(), worker_id),
        )
        c.commit()
        ok = cur.rowcount > 0
    if ok and (current_job_id or "").strip():
        train_jobs.heartbeat_job((current_job_id or "").strip(), worker_id)
    return ok


def job_finished(worker_id: str) -> bool:
    """Worker reports its current job done. Clears the in-progress slot."""
    return heartbeat(worker_id, "")


def _alive_workers() -> list[sqlite3.Row]:
    c = _connect()
    cutoff = time.time() - HEARTBEAT_TIMEOUT_S
    return c.execute(
        "SELECT id, platform, last_seen, current_job_id FROM forge_workers"
        " WHERE dead=0 AND last_seen >= ?",
        (cutoff,),
    ).fetchall()


def status_snapshot() -> dict:
    """Dispatcher status: alive workers per platform, dead count, queue depth.

    Also returns the per-worker list (id, platform, last heartbeat, current
    job, alive flag) — the aggregates alone can't tell ops WHICH worker died
    or what job it was stuck on. Each worker also carries a `status`
    ("alive" / "stale" / "dead") and `seconds_since_heartbeat`, and the
    snapshot carries a top-level `stale` count: a worker past the heartbeat
    timeout but not yet reaped is "stale", not dead — it still holds its
    current job, and ops needs to see it before hitting "reap dead".
    """
    from . import train as train_jobs

    alive = _alive_workers()
    per_platform: dict[str, int] = {}
    for w in alive:
        per_platform[w["platform"]] = per_platform.get(w["platform"], 0) + 1
    c = _connect()
    dead = c.execute("SELECT COUNT(*) FROM forge_workers WHERE dead=1").fetchone()[0]
    now = time.time()
    cutoff = now - HEARTBEAT_TIMEOUT_S
    workers = []
    stale = 0
    for r in c.execute(
        "SELECT id, platform, last_seen, current_job_id, dead "
        "FROM forge_workers ORDER BY last_seen DESC"
    ).fetchall():
        is_alive = bool(not r["dead"] and r["last_seen"] >= cutoff)
        if r["dead"]:
            status = "dead"
        elif is_alive:
            status = "alive"
        else:
            status = "stale"
            stale += 1
        workers.append(
            {
                "id": r["id"],
                "platform": r["platform"],
                "last_seen": r["last_seen"],
                "seconds_since_heartbeat": max(0, int(now - r["last_seen"])),
                "status": status,
                "current_job_id": r["current_job_id"] or None,
                "alive": is_alive,
            }
        )
    return {
        "alive": len(alive),
        "stale": stale,
        "dead": int(dead),
        "platforms": per_platform,
        "queued_jobs": train_jobs.count_queued_jobs(),
        "workers": workers,
    }


def reap_dead_workers() -> dict:
    """Mark workers that missed heartbeats as dead; re-queue their stuck jobs.

    Returns {"reaped": <workers marked dead>,
             "requeued": [job ids put back to 'queued'],
             "failed": [job ids failed at the attempts cap]}. Jobs not in
    'working' are never touched. A poison job whose worker keeps dying
    reaches the attempts cap and is failed by requeue_job() instead of
    requeued — before this fix the reaper requeued it forever, because
    requeue_job() never bumped attempts and claim_job() never checked
    them, so each reap just handed the same doomed job to the next
    worker."""
    from . import train as train_jobs

    c = _connect()
    cutoff = time.time() - HEARTBEAT_TIMEOUT_S
    with _lock:
        stuck = c.execute(
            "SELECT id, current_job_id FROM forge_workers WHERE dead=0 AND last_seen < ?",
            (cutoff,),
        ).fetchall()
        requeued: list[str] = []
        failed: list[str] = []
        for w in stuck:
            jid = w["current_job_id"] or ""
            if jid:
                if train_jobs.requeue_job(jid):
                    requeued.append(jid)
                elif (train_jobs.get_job(jid) or {}).get("status") == "failed":
                    failed.append(jid)
            c.execute(
                "UPDATE forge_workers SET dead=1, current_job_id='' WHERE id=?",
                (w["id"],),
            )
        c.commit()
    return {"reaped": len(stuck), "requeued": requeued, "failed": failed}
