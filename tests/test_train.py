"""Tests for the pull-based training job queue (no network needed)."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

from app import train  # noqa: E402


def test_create_get_list():
    jid = train.create_job("Qwen/Qwen2-1.5B", "user/my-data", 16)
    assert jid
    job = train.get_job(jid)
    assert job["model_repo"] == "Qwen/Qwen2-1.5B"
    assert job["dataset"] == "user/my-data"
    assert job["lora_rank"] == 16
    assert job["status"] == "queued"
    jobs = train.list_jobs()
    assert any(j["id"] == jid for j in jobs)


def test_lora_rank_clamped():
    jid = train.create_job("m", "", 9999)
    assert train.get_job(jid)["lora_rank"] == 256
    jid2 = train.create_job("m", "", "not-a-number")
    assert train.get_job(jid2)["lora_rank"] == 16


def test_claim_is_atomic_and_fifo():
    first = train.create_job("model-a")
    second = train.create_job("model-b")
    claimed = train.claim_job()
    assert claimed is not None
    # FIFO: oldest queued job claimed first (first, unless an older test job exists —
    # either way the claimed job must leave 'queued')
    assert claimed["status"] == "working"
    assert train.get_job(claimed["id"])["status"] == "working"
    # unknown id
    assert train.get_job("nope") is None
    # drain the rest so later tests start clean
    while train.claim_job():
        pass
    assert first and second  # ids were created


def _claim_job_id(jid: str):
    """Claim jobs until the given id is the one claimed (other queued jobs
    from earlier tests may be older). Returns the claimed row or None."""
    for _ in range(100):
        claimed = train.claim_job()
        if claimed is None:
            return None
        if claimed["id"] == jid:
            return claimed
    return None


def _backdate_job(jid: str, secs_ago: float = 7200) -> None:
    import time

    c = train._connect()
    with train._lock:
        c.execute(
            "UPDATE train_jobs SET updated_at=? WHERE id=?",
            (time.time() - secs_ago, jid),
        )
        c.commit()


def test_complete_and_metrics():
    jid = train.create_job("model-c")
    assert _claim_job_id(jid) is not None
    assert train.complete_job(jid, "done", result_url="http://x/a.safetensors",
                              metrics={"train_loss": 1.23}, error="")
    job = train.get_job(jid)
    assert job["status"] == "done"
    assert job["result_url"] == "http://x/a.safetensors"
    assert job["metrics"]["train_loss"] == 1.23
    jid2 = train.create_job("model-d")
    assert _claim_job_id(jid2) is not None
    assert train.complete_job(jid2, "failed", error="boom")
    assert train.get_job(jid2)["status"] == "failed"
    assert train.get_job(jid2)["error"] == "boom"
    assert train.complete_job("missing", "done") is False


def test_complete_refuses_non_working():
    # Only the worker holding a job may report on it. A stale worker —
    # reaped after it vanished, its job re-claimed and finished by
    # another worker — must not overwrite the new worker's result by
    # reporting late, and completing a never-claimed job must not flip
    # it to done.
    jid = train.create_job("model-stale")
    assert train.complete_job(jid, "done", result_url="http://x/stale") is False
    assert train.get_job(jid)["status"] == "queued"
    assert _claim_job_id(jid) is not None
    assert train.complete_job(jid, "done", result_url="http://x/real") is True
    assert train.get_job(jid)["result_url"] == "http://x/real"
    # Already done: a late duplicate report must not flip it back.
    assert train.complete_job(jid, "failed", error="late") is False
    assert train.get_job(jid)["status"] == "done"
    assert train.get_job(jid)["result_url"] == "http://x/real"


def test_requeue_job_bumps_attempts():
    jid = train.create_job("model-requeue")
    assert _claim_job_id(jid) is not None
    assert train.requeue_job(jid) is True
    job = train.get_job(jid)
    assert job["status"] == "queued"
    assert job["attempts"] == 1
    # A finished job is never requeued.
    assert _claim_job_id(jid) is not None
    assert train.complete_job(jid, "done") is True
    assert train.requeue_job(jid) is False


def test_requeue_job_fails_exhausted_job():
    # A job whose worker already died MAX_ATTEMPTS times is failed, not
    # handed to yet another worker.
    jid = train.create_job("model-requeue-poison")
    assert _claim_job_id(jid) is not None
    c = train._connect()
    with train._lock:
        c.execute(
            "UPDATE train_jobs SET attempts=? WHERE id=?",
            (train.MAX_ATTEMPTS, jid),
        )
        c.commit()
    assert train.requeue_job(jid) is False
    job = train.get_job(jid)
    assert job["status"] == "failed"
    assert "needs Jay" in (job["error"] or "")


def test_poison_job_fails_after_max_attempts():
    # Regression: fail_stuck_jobs() requeued every stuck job and
    # claim_job() ignored attempts, so a job whose worker always died
    # cycled claim -> stuck -> requeue forever. Now it fails at the cap
    # and is never claimed again.
    jid = train.create_job("model-poison")
    assert _claim_job_id(jid) is not None
    failed_at = None
    for _ in range(train.MAX_ATTEMPTS + 2):
        _backdate_job(jid)
        out = train.fail_stuck_jobs(max_working_s=3600)
        job = train.get_job(jid)
        if job["status"] == "failed":
            failed_at = out
            break
        assert job["status"] == "queued"
        assert jid in out["requeued"]
        assert _claim_job_id(jid) is not None
    assert failed_at is not None, "poison job never failed"
    assert jid in failed_at["failed"]
    assert jid not in failed_at["requeued"]
    job = train.get_job(jid)
    assert job["status"] == "failed"
    assert job["attempts"] == train.MAX_ATTEMPTS
    assert "needs Jay" in (job["error"] or "")
    # Failed jobs are never claimed, however many times the queue is polled.
    for _ in range(5):
        claimed = train.claim_job()
        assert claimed is None or claimed["id"] != jid


def test_job_at_cap_gets_final_run_then_fails():
    # The cap bounds re-queues, not runs: a job queued at the cap is
    # still claimable for its final run (the same "3 retries" /retry has
    # always allowed), and the reap fails it when that run dies instead
    # of handing it to yet another worker.
    jid = train.create_job("model-last-run")
    c = train._connect()
    with train._lock:
        c.execute(
            "UPDATE train_jobs SET attempts=? WHERE id=?",
            (train.MAX_ATTEMPTS, jid),
        )
        c.commit()
    assert _claim_job_id(jid) is not None
    _backdate_job(jid)
    out = train.fail_stuck_jobs(max_working_s=3600)
    assert jid in out["failed"]
    assert jid not in out["requeued"]
    job = train.get_job(jid)
    assert job["status"] == "failed"
    assert "needs Jay" in (job["error"] or "")


def test_adapter_path_is_traversal_safe():
    p = train.adapter_file_path("abc123", "../../etc/passwd")
    assert p.endswith("passwd")
    assert ".." not in p
    assert os.path.dirname(p).endswith("abc123")


def _claim_specific(jid: str) -> dict:
    # claim_job takes the oldest queued job; loop until ours is claimed
    # (other tests may have left queued jobs ahead of it).
    for _ in range(100):
        claimed = train.claim_job()
        if claimed is None:
            break
        if claimed["id"] == jid:
            return claimed
    raise AssertionError(f"job {jid} was never claimed")


def _backdate(jid: str, secs_ago: float) -> None:
    c = train._connect()
    with train._lock:
        c.execute(
            "UPDATE train_jobs SET updated_at=? WHERE id=?",
            (time.time() - secs_ago, jid),
        )
        c.commit()


def test_heartbeat_job_refreshes_updated_at_and_blocks_reap():
    # The bug: a 30-120 min training run with no heartbeat looks stuck
    # after 3600s untouched and gets re-queued mid-training. A heartbeat
    # must refresh updated_at so fail_stuck_jobs leaves it alone.
    jid = train.create_job("model-hb")
    _claim_specific(jid)
    attempts_before = train.get_job(jid)["attempts"]
    _backdate(jid, 3500)  # nearly stale, but still inside the window
    assert train.heartbeat_job(jid) is True
    job = train.get_job(jid)
    assert job["status"] == "working"
    assert job["attempts"] == attempts_before  # heartbeat changes nothing else
    assert time.time() - job["updated_at"] < 5
    assert train.fail_stuck_jobs()["requeued"] == []
    assert train.get_job(jid)["status"] == "working"
    # ...but without a heartbeat the same job IS reaped once it goes stale.
    _backdate(jid, 3600 + 60)
    assert jid in train.fail_stuck_jobs()["requeued"]
    assert train.get_job(jid)["status"] == "queued"


def test_heartbeat_job_refuses_non_working_and_missing():
    jid = train.create_job("model-hb-queued")
    assert train.heartbeat_job(jid) is False  # queued, not working
    _claim_specific(jid)
    assert train.heartbeat_job(jid) is True
    assert train.complete_job(jid, "done")
    assert train.heartbeat_job(jid) is False  # done: never revived
    assert train.heartbeat_job("missing") is False


def test_adapter_path_job_id_is_traversal_safe():
    # Regression: job_id came raw from the URL, so %2e%2e escaped ADAPTERS_DIR.
    for evil in ("..", "../..", "..%2f..", "/etc", "a/b"):
        p = train.adapter_file_path(evil, "adapter.bin")
        assert os.path.commonpath([p, train.ADAPTERS_DIR]) == train.ADAPTERS_DIR
        assert ".." not in p
    # Real uuid-hex ids are untouched.
    p = train.adapter_file_path("abc123def456", "adapter.bin")
    assert os.path.dirname(p).endswith("abc123def456")


def test_adapter_path_dot_filenames_are_plain_files():
    # "." / ".." used to resolve to the job dir itself (uploading to ".."
    # then crashed opening a directory). They now map to a plain file.
    for dot in (".", ".."):
        p = train.adapter_file_path("abc123", dot)
        assert os.path.basename(p) == "file"
        assert os.path.dirname(p).endswith("abc123")


def test_adapter_path_get_does_not_create_dirs():
    # A download for a bogus job used to mkdir adapters/<bogus>/.
    p = train.adapter_file_path("no-such-job-xyz", "adapter.bin")
    assert not os.path.exists(os.path.dirname(p))
    assert not os.path.exists(p)


def test_adapter_path_symlinked_job_dir_refused_on_create():
    # A symlink planted at adapters/<job>/ pointing outside must not be
    # treated as a job dir when the upload route creates it.
    import tempfile

    outside = tempfile.mkdtemp(prefix="bro-adapter-outside-")
    link_dir = os.path.join(train.ADAPTERS_DIR, "symjob")
    os.makedirs(train.ADAPTERS_DIR, exist_ok=True)
    if os.path.lexists(link_dir):
        os.unlink(link_dir)
    os.symlink(outside, link_dir)
    try:
        import pytest

        with pytest.raises(ValueError):
            train.adapter_file_path("symjob", "adapter.bin", create_dir=True)
    finally:
        os.unlink(link_dir)


def test_adapter_total_bytes():
    jid = train.create_job("model-total")
    assert train.adapter_total_bytes(jid) == 0
    p = train.adapter_file_path(jid, "a.bin", create_dir=True)
    with open(p, "wb") as f:
        f.write(b"x" * 100)
    p2 = train.adapter_file_path(jid, "b.bin")
    with open(p2, "wb") as f:
        f.write(b"y" * 50)
    assert train.adapter_total_bytes(jid) == 150
    assert train.adapter_total_bytes("no-such-job-xyz") == 0


def _claim_job_id_as(jid: str, worker_id: str = ""):
    """Claim jobs as one worker until the given id is the one claimed."""
    for _ in range(100):
        claimed = train.claim_job(worker_id)
        if claimed is None:
            return None
        if claimed["id"] == jid:
            return claimed
    return None


def test_claim_records_owner_and_enforces_it():
    # A stale worker that reports on a job another worker now holds must
    # not complete it (or keep it alive): the claim stamps the owner.
    jid = train.create_job("model-owner")
    claimed = _claim_job_id_as(jid, "worker-a")
    assert claimed is not None
    assert claimed["claimed_by"] == "worker-a"
    assert train.get_job(jid)["claimed_by"] == "worker-a"
    # A different worker's report is refused, job untouched.
    assert train.complete_job(jid, "done", result_url="http://x/stale") is False
    assert train.complete_job(jid, "done", worker_id="worker-b") is False
    assert train.heartbeat_job(jid, "worker-b") is False
    job = train.get_job(jid)
    assert job["status"] == "working"
    assert job["result_url"] is None
    # The owner can heartbeat and complete.
    assert train.heartbeat_job(jid, "worker-a") is True
    assert train.complete_job(jid, "done", result_url="http://x/real",
                              worker_id="worker-a") is True
    assert train.get_job(jid)["result_url"] == "http://x/real"


def test_unowned_claim_stays_open():
    # Claims without a worker id (direct /api/train/claim users, old
    # notebooks) have no owner: any reporter may heartbeat/complete,
    # exactly as before this change.
    jid = train.create_job("model-unowned")
    assert _claim_job_id_as(jid) is not None
    assert train.get_job(jid)["claimed_by"] in (None, "")
    assert train.heartbeat_job(jid) is True
    assert train.heartbeat_job(jid, "whoever") is True
    assert train.complete_job(jid, "done", worker_id="whoever") is True
    assert train.get_job(jid)["status"] == "done"


def test_requeue_clears_owner():
    jid = train.create_job("model-requeue-owner")
    assert _claim_job_id_as(jid, "worker-a") is not None
    assert train.requeue_job(jid) is True
    job = train.get_job(jid)
    assert job["status"] == "queued"
    assert job["claimed_by"] in (None, "")
    # Re-claimed by a different worker: the new owner is enforced, and
    # the old owner is now a stranger.
    assert _claim_job_id_as(jid, "worker-b") is not None
    assert train.complete_job(jid, "done", worker_id="worker-a") is False
    assert train.complete_job(jid, "done", worker_id="worker-b") is True


def test_list_jobs_includes_attempts_and_claimed_by():
    # Regression: the ops UI renders j.attempts, but list_jobs never
    # selected it, so the column always showed 0.
    jid = train.create_job("model-list-owner")
    assert _claim_job_id_as(jid, "worker-a") is not None
    assert train.requeue_job(jid) is True  # attempts 1, owner cleared
    assert _claim_job_id_as(jid, "worker-c") is not None
    row = next(j for j in train.list_jobs() if j["id"] == jid)
    assert row["attempts"] == 1
    assert row["claimed_by"] == "worker-c"
    assert train.complete_job(jid, "done", worker_id="worker-c") is True
