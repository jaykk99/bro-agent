"""Pull-based GPU training job queue — SQLite-backed, inside bro-agent.

The bro app queues fine-tune jobs; a Colab/Kaggle worker
(notebooks/train-worker.ipynb) claims them over HTTP, trains with Unsloth
QLoRA, uploads the LoRA adapter back to bro-agent, and wipes its disk.
Same pull system as the Forge Asset Pipeline, but fully inside bro-agent:
no Supabase, no Error Inbox, no third-party host.

Auth: every HTTP route goes through the app's access-code gate in main.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sqlite3
import stat
import threading
import time
import uuid

from .memory import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(DB_PATH)), "adapters")

# A job whose worker keeps dying is poison: bad model repo, OOM, poison
# dataset. Every re-queue path (reap-stuck, forge reaper, manual /retry)
# bumps attempts, and a job at the cap is failed instead of re-queued —
# at most MAX_ATTEMPTS re-queues, so a poison job gets MAX_ATTEMPTS + 1
# runs and then retires instead of burning GPU worker slots forever.
# The cap is per re-queue, not per run: a job queued at the cap is still
# claimable for its final run (that is how /retry's "3 retries" has
# always worked), and the reap fails it when that run dies. Before this
# fix only retry_job checked the cap, so a poison job cycled
# claim -> stuck -> requeue -> claim without end through the automatic
# paths — the "attempts cap stops poison jobs" claim in
# TRAIN_PIPELINE.md was only true for the manual /retry endpoint.
MAX_ATTEMPTS = 3


def _exhausted_error(attempts: int) -> str:
    return (
        f"worker died {attempts} times (max {MAX_ATTEMPTS} attempts); "
        "needs Jay"
    )


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
    safe across processes/workers.

    No attempts check here, deliberately: the cap bounds re-queues, not
    runs, so a job queued at the cap is claimable for its final run (the
    same "3 retries" the /retry endpoint has always allowed). The reap
    paths fail the job when that run dies, so it can never cycle again."""
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
    """Mark a job done/failed. Only a job currently 'working' can complete:
    the worker that claimed it is the only one whose result counts. A
    stale worker — reaped after it vanished, its job re-claimed and finished
    by another worker — used to overwrite the new worker's result (or flip
    an already-done job back to failed) just by reporting late. Returns
    False when the job is missing or not 'working'."""
    if status not in ("done", "failed"):
        raise ValueError("status must be done or failed")
    c = _connect()
    with _lock:
        cur = c.execute(
            "UPDATE train_jobs SET status=?, result_url=?, metrics_json=?,"
            " error=?, updated_at=? WHERE id=? AND status='working'",
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
    if the job is not currently 'working' (finished jobs are never touched).

    Attempts are bumped here, not just in fail_stuck_jobs: the forge
    reaper calls this directly, and without the bump a poison job would
    requeue forever through that path too. A job already at the attempts
    cap is failed instead (its worker kept dying) and this returns False,
    so callers that report requeued ids never list it as requeued."""
    c = _connect()
    with _lock:
        row = c.execute(
            "SELECT status, attempts FROM train_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if not row or row["status"] != "working":
            return False
        attempts = row["attempts"] or 0
        if attempts >= MAX_ATTEMPTS:
            c.execute(
                "UPDATE train_jobs SET status='failed', error=?, updated_at=?"
                " WHERE id=?",
                (_exhausted_error(attempts), time.time(), job_id),
            )
            c.commit()
            return False
        cur = c.execute(
            "UPDATE train_jobs SET status='queued', attempts=attempts+1,"
            " updated_at=? WHERE id=? AND status='working'",
            (time.time(), job_id),
        )
        c.commit()
        return cur.rowcount > 0


def retry_job(job_id: str, max_attempts: int = MAX_ATTEMPTS) -> bool:
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


def fail_stuck_jobs(max_working_s: float = 3600) -> dict:
    """Re-queue 'working' jobs whose worker vanished without completing.

    The forge dispatcher reaps dead *workers*; this covers jobs stuck in
    'working' with no worker row at all (e.g. claimed via /api/train/claim
    directly, then the notebook died).

    Returns {"requeued": [ids put back to 'queued'],
             "failed": [ids failed at the attempts cap]}. A stuck job at
    the cap is FAILED, not requeued: before this fix every stuck job was
    requeued with attempts+1 and claim_job() ignored attempts entirely,
    so a poison job cycled claim -> stuck -> requeue forever — the
    "attempts cap stops poison jobs" claim in TRAIN_PIPELINE.md was only
    true for the manual /retry endpoint, never for this automatic path.
    Attempts are bumped on requeue so a job that keeps dying reaches the
    cap and retires instead of spinning."""
    c = _connect()
    cutoff = time.time() - max_working_s
    requeued: list[str] = []
    failed: list[str] = []
    with _lock:
        stuck = c.execute(
            "SELECT id, attempts FROM train_jobs WHERE status='working' AND updated_at < ?",
            (cutoff,),
        ).fetchall()
        for r in stuck:
            attempts = r["attempts"] or 0
            if attempts >= MAX_ATTEMPTS:
                c.execute(
                    "UPDATE train_jobs SET status='failed', error=?,"
                    " updated_at=? WHERE id=?",
                    (_exhausted_error(attempts), time.time(), r["id"]),
                )
                failed.append(r["id"])
            else:
                c.execute(
                    "UPDATE train_jobs SET status='queued', attempts=attempts+1,"
                    " updated_at=? WHERE id=? AND status='working'",
                    (time.time(), r["id"]),
                )
                requeued.append(r["id"])
        c.commit()
    return {"requeued": requeued, "failed": failed}


def count_queued_jobs() -> int:
    """Queue depth for the forge dispatcher status snapshot."""
    c = _connect()
    row = c.execute("SELECT COUNT(*) FROM train_jobs WHERE status='queued'").fetchone()
    return int(row[0]) if row else 0


def adapter_job_dir(job_id: str) -> str:
    """Sanitized on-disk directory for one job's adapter files (no creation)."""
    safe_job = re.sub(r"[^A-Za-z0-9_-]", "_", job_id or "")[:64] or "job"
    return os.path.join(ADAPTERS_DIR, safe_job)


_job_upload_locks: dict[str, asyncio.Lock] = {}
_job_upload_locks_guard = threading.Lock()


@contextlib.asynccontextmanager
async def adapter_upload_lock(job_id: str):
    """Serialize uploads to one job (different jobs still run in parallel).

    The per-job cap is enforced check-then-stream: the route snapshots
    the job's stored bytes, streams against that snapshot, then replaces
    the file. With no lock, two concurrent uploads to one job both
    snapshot before either lands and both fit on their own numbers, so
    the job ends up over its cap (proven 2026-11-01: two 20 MB uploads
    against a 30 MB cap both returned 200 and stored 40 MB). Holding this
    lock across snapshot, stream, and replace makes the second upload see
    the first one's landed bytes and 413 like a sequential upload would.

    This must be an asyncio lock, not a threading lock: the upload route
    is async and awaits file reads while holding it, and a threading lock
    would block the event-loop thread on the second acquirer and deadlock
    the first upload's progress. The registry guard is a threading lock
    only because it is held for a dict lookup with no await inside it.
    """
    key = os.path.basename(adapter_job_dir(job_id))
    with _job_upload_locks_guard:
        lock = _job_upload_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _job_upload_locks[key] = lock
    async with lock:
        yield


def adapters_total_bytes() -> int:
    """Bytes stored for ALL jobs (lstat sizes, never a symlink target's).

    This is the committed-files snapshot for the aggregate cap. It
    deliberately skips in-flight `.upload-*.tmp` files: those bytes are
    tracked by the upload reservation ledger instead, so counting them
    here too would double-count every concurrent upload. A symlinked
    job dir is skipped whole — its contents are not this tree's files.
    Missing adapters dir = 0."""
    total = 0
    try:
        with os.scandir(ADAPTERS_DIR) as job_dirs:
            job_dirs = list(job_dirs)
    except OSError:
        return 0
    for job_dir in job_dirs:
        try:
            if not job_dir.is_dir(follow_symlinks=False):
                continue
            with os.scandir(job_dir.path) as entries:
                for entry in entries:
                    try:
                        if entry.name.startswith(".upload-") and entry.name.endswith(".tmp"):
                            continue
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


# In-flight upload reservations for the aggregate cap. The per-job lock
# serializes uploads to one job, but uploads to DIFFERENT jobs run
# concurrently, so a snapshot-then-stream aggregate check would let two
# jobs each fit on their own numbers and land over the total together —
# the same race the per-job lock fixed for one job. Each upload instead
# reserves the bytes it has streamed so far; the aggregate check reads
# the committed snapshot plus every live reservation, so a concurrent
# upload's not-yet-landed bytes already count against the cap. The
# ledger is a plain dict under a threading lock: updates happen between
# awaits, never across one, so it is safe to touch from the async route
# and cannot deadlock the event loop.
_inflight_reserved: dict[str, int] = {}
_inflight_lock = threading.Lock()


def reserve_upload(token: str, size: int) -> None:
    """Record how many bytes one in-flight upload has streamed so far."""
    with _inflight_lock:
        _inflight_reserved[token] = max(0, size)


def release_upload(token: str) -> None:
    """Drop one upload's reservation (landed, rejected, or aborted)."""
    with _inflight_lock:
        _inflight_reserved.pop(token, None)


def inflight_reserved_bytes() -> int:
    """Bytes currently reserved by all in-flight uploads."""
    with _inflight_lock:
        return sum(_inflight_reserved.values())


_commit_lock = threading.Lock()


@contextlib.contextmanager
def adapter_commit_lock():
    """Serialize the final cap re-check + replace across jobs.

    The per-chunk aggregate check reads a snapshot taken at upload
    start, so an upload that lands and releases mid-stream would go
    invisible to it for the rest of the stream. The final re-check
    under this lock re-scans committed bytes just before the replace,
    so two jobs racing the total cap cannot both pass and both land.
    Held only for a scan plus a replace — never across streaming — so
    uploads still run in parallel; only their final instant serializes.
    A threading lock is safe here: no await happens inside it."""
    with _commit_lock:
        yield


def adapter_total_bytes(job_id: str) -> int:
    """Bytes already stored for one job (lstat sizes, so a symlink entry
    counts only its own link size, never its target's). Missing dir = 0."""
    d = adapter_job_dir(job_id)
    total = 0
    try:
        with os.scandir(d) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def adapter_file_path(job_id: str, filename: str, create_dir: bool = False) -> str:
    """Sanitized on-disk path for an uploaded adapter file (path traversal safe).

    Both segments are sanitized: the filename was already basenamed, but the
    job_id came straight from the URL path, so GET /api/train/%2e%2e/files/x
    used to resolve outside ADAPTERS_DIR and serve arbitrary files next to it.
    Real job ids are uuid hex, so the allowlist below changes nothing legit.

    Hardening, 2026-10-30 (the per-file 2 GB cap from 2026-10-29 did not
    bound the job, and this path never jailed symlinks):
      * dot-only filenames ("." / "..") used to resolve to the job dir
        itself (an upload to ".." then crashed opening a directory); they
        now map to a plain file, like any other bad name.
      * the path is NOT created by default: a GET for a bogus job used to
        mkdir adapters/<bogus>/ as a side effect, so anyone with the access
        code could litter the disk with directories. Only the upload route
        passes create_dir=True.
      * when creating, the real job dir must stay inside ADAPTERS_DIR —
        a symlink planted at adapters/<job>/ used to make every later
        upload overwrite, and every download serve, the link target
        outside the adapters tree (proven with a planted link before this
        fix). A symlinked job dir raises ValueError here; the routes also
        refuse a symlinked file itself.
    """
    safe_job = re.sub(r"[^A-Za-z0-9_-]", "_", job_id or "")[:64] or "job"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(filename or ""))[:128]
    if safe in ("", ".", ".."):
        safe = "file"
    d = os.path.join(ADAPTERS_DIR, safe_job)
    if create_dir:
        os.makedirs(d, exist_ok=True)
        real_root = os.path.realpath(ADAPTERS_DIR)
        real_dir = os.path.realpath(d)
        if os.path.commonpath([real_dir, real_root]) != real_root:
            raise ValueError("adapter job dir escapes the adapters directory")
    return os.path.join(d, safe)


def open_adapter_job_dir(job_id: str) -> int:
    """Open one job's adapter dir pinned to its inode (no TOCTOU).

    The upload route used to create its temp file and os.replace() it
    into place by path: both operations resolved the job-dir path, so
    a symlink swapped in after adapter_file_path()'s realpath check
    (or after the route's pre-replace symlink re-check) made the temp
    file — and then the landed adapter — resolve through the link to a
    target outside the adapters tree. The download side was already
    pinned this way (open_adapter_file); the upload side was not, and
    the pre-replace check could not close it: it only inspected the
    path at one instant, while the create and the replace each
    re-resolved it afterwards.

    This opens the dir itself with O_DIRECTORY | O_NOFOLLOW, so a
    symlinked job dir fails here instead of being followed, and the
    caller then creates, writes, and renames strictly relative to the
    returned fd — a path swap after this open cannot redirect any of
    those operations, because none of them re-resolves the path. The
    result must be a directory, and on Linux the /proc/self/fd link is
    re-resolved and must still sit inside the adapters tree, covering
    a symlinked adapters root itself (same rule as open_adapter_file).

    Returns the dir fd. The caller owns it and must close it.
    Raises FileNotFoundError for a missing dir, a symlink anywhere in
    the path, a non-directory, or a resolved target outside the tree.
    """
    d = adapter_job_dir(job_id)
    real_root = os.path.realpath(ADAPTERS_DIR)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        dir_fd = os.open(
            d,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow,
        )
    except OSError as exc:
        raise FileNotFoundError("adapter job dir not found") from exc
    try:
        st = os.fstat(dir_fd)
        if not stat.S_ISDIR(st.st_mode):
            raise FileNotFoundError("adapter job dir is not a directory")
        try:
            resolved = os.readlink(f"/proc/self/fd/{dir_fd}")
        except OSError:
            resolved = None
        if resolved is not None and os.path.commonpath(
            [os.path.realpath(resolved), real_root]
        ) != real_root:
            raise FileNotFoundError("adapter job dir escapes the adapters directory")
    except BaseException:
        os.close(dir_fd)
        raise
    return dir_fd


def open_adapter_file(job_id: str, filename: str) -> tuple[int, int, str]:
    """Open one adapter file for download, pinned to its inode (no TOCTOU).

    The download route used to check islink/isfile on the path and then
    hand that path to FileResponse, which opened it later: a symlink
    swapped in between the check and the open made the route serve the
    link target outside the adapters tree. This opens the file itself
    and returns its descriptor, so the caller streams from the inode —
    a path swap after the open cannot change what is served, and an
    os.replace() mid-download serves the complete old adapter instead
    of a half-written new one (matching the upload route's atomicity).

    How the open refuses symlinks, with no check-then-open gap:
      * the job dir is opened with O_DIRECTORY | O_NOFOLLOW, so a
        symlinked job dir fails here instead of being followed;
      * the file is opened by name relative to that dir fd, again with
        O_NOFOLLOW, so a symlinked file fails instead of being followed;
      * the result must be a regular file (fstat, not a path check);
      * on Linux the /proc/self/fd link is re-resolved and must still
        sit inside the adapters tree, covering a symlinked adapters
        root itself. Platforms without /proc keep the pre-open realpath
        check below plus the two O_NOFOLLOW opens, which is no worse
        than the old route.

    Returns (fd, size_bytes, safe_name). The caller owns the fd and
    must close it. Raises FileNotFoundError for a missing file/dir, a
    symlink anywhere in the path, a non-regular file, an in-flight
    `.upload-*.tmp` name, or a resolved target outside the tree.
    """
    if os.path.basename(filename or "").startswith(".upload-"):
        raise FileNotFoundError("in-flight upload temp files are never servable")
    dest = adapter_file_path(job_id, filename)
    job_dir, safe_name = os.path.dirname(dest), os.path.basename(dest)
    # Pre-open realpath check: the same rule the old route applied after
    # its symlink check, kept for platforms without /proc.
    real_root = os.path.realpath(ADAPTERS_DIR)
    if os.path.commonpath([os.path.realpath(dest), real_root]) != real_root:
        raise FileNotFoundError("adapter file escapes the adapters directory")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        dir_fd = os.open(
            job_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow,
        )
    except OSError as exc:
        raise FileNotFoundError("adapter job dir not found") from exc
    try:
        fd = os.open(safe_name, os.O_RDONLY | nofollow, dir_fd=dir_fd)
    except OSError as exc:
        raise FileNotFoundError("adapter file not found") from exc
    finally:
        os.close(dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise FileNotFoundError("adapter file is not a regular file")
        try:
            resolved = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            resolved = None
        if resolved is not None and os.path.commonpath(
            [os.path.realpath(resolved), real_root]
        ) != real_root:
            raise FileNotFoundError("adapter file escapes the adapters directory")
    except BaseException:
        os.close(fd)
        raise
    return fd, st.st_size, safe_name
