"""Tests for the forge worker dispatcher (no network needed)."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-dispatch-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

from app import dispatch  # noqa: E402
from app import train  # noqa: E402


def _backdate_worker(wid: str, secs_ago: float) -> None:
    c = dispatch._connect()
    with dispatch._lock:
        c.execute(
            "UPDATE forge_workers SET last_seen=? WHERE id=?",
            (time.time() - secs_ago, wid),
        )
        c.commit()


def test_register_heartbeat_finish():
    wid = dispatch.register_worker("colab")
    assert wid
    assert dispatch.heartbeat(wid, "") is True
    assert dispatch.heartbeat("no-such-worker") is False
    assert dispatch.job_finished(wid) is True
    snap = dispatch.status_snapshot()
    assert snap["alive"] >= 1
    assert snap["platforms"].get("colab", 0) >= 1
    assert "queued_jobs" in snap


def test_platform_normalized():
    wid = dispatch.register_worker("some-random-platform")
    snap = dispatch.status_snapshot()
    assert snap["platforms"].get("other", 0) >= 1
    dispatch.job_finished(wid)


def test_reap_dead_worker_requeues_stuck_job():
    jid = train.create_job("model-x")
    claimed = None
    for _ in range(100):
        c = train.claim_job()
        if c is None:
            break
        if c["id"] == jid:
            claimed = c
            break
    assert claimed is not None and claimed["status"] == "working"
    wid = dispatch.register_worker("kaggle")
    assert dispatch.heartbeat(wid, jid) is True
    # simulate the worker dying: last heartbeat older than the timeout
    _backdate_worker(wid, dispatch.HEARTBEAT_TIMEOUT_S + 60)
    out = dispatch.reap_dead_workers()
    assert out["reaped"] >= 1
    assert jid in out["requeued"]
    assert train.get_job(jid)["status"] == "queued"
    snap = dispatch.status_snapshot()
    assert snap["dead"] >= 1


def test_reap_skips_jobs_not_working():
    jid = train.create_job("model-y")  # stays queued, not claimed
    wid = dispatch.register_worker("colab")
    dispatch.heartbeat(wid, jid)
    _backdate_worker(wid, dispatch.HEARTBEAT_TIMEOUT_S + 60)
    out = dispatch.reap_dead_workers()
    assert jid not in out["requeued"]
    assert train.get_job(jid)["status"] == "queued"


def test_snapshot_lists_workers_with_alive_flags():
    wid_alive = dispatch.register_worker("colab")
    dispatch.heartbeat(wid_alive, "job-live")
    wid_dead = dispatch.register_worker("kaggle")
    dispatch.heartbeat(wid_dead, "job-gone")
    _backdate_worker(wid_dead, dispatch.HEARTBEAT_TIMEOUT_S + 60)
    dispatch.reap_dead_workers()
    snap = dispatch.status_snapshot()
    by_id = {w["id"]: w for w in snap["workers"]}
    assert by_id[wid_alive]["alive"] is True
    assert by_id[wid_alive]["current_job_id"] == "job-live"
    assert by_id[wid_alive]["platform"] == "colab"
    assert by_id[wid_dead]["alive"] is False
    assert by_id[wid_dead]["current_job_id"] is None  # cleared on reap


def test_reap_idempotent():
    wid = dispatch.register_worker("colab")
    _backdate_worker(wid, dispatch.HEARTBEAT_TIMEOUT_S + 60)
    first = dispatch.reap_dead_workers()
    second = dispatch.reap_dead_workers()
    assert first["reaped"] >= 1
    assert second["reaped"] == 0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("ALL DISPATCH TESTS PASS")
