"""Tests for the Forge worker dispatcher (no network needed)."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-dispatch-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

from app import dispatch  # noqa: E402


def test_register_and_heartbeat():
    w1 = dispatch.register_worker("colab")
    w2 = dispatch.register_worker("kaggle")
    assert w1 and w2 and w1 != w2
    assert dispatch.heartbeat(w1, "job-123") is True
    assert dispatch.heartbeat("nope") is False
    ws = {w["worker_id"]: w for w in dispatch.list_workers()}
    assert ws[w1]["alive"] is True
    assert ws[w1]["current_job_id"] == "job-123"
    assert ws[w1]["platform"] == "colab"
    assert ws[w2]["platform"] == "kaggle"


def test_platform_normalized():
    w = dispatch.register_worker("weird-platform")
    ws = {w2["worker_id"]: w2 for w2 in dispatch.list_workers()}
    assert ws[w]["platform"] == "unknown"


def test_job_finished_counts_and_clears():
    w = dispatch.register_worker("colab")
    dispatch.heartbeat(w, "job-9")
    dispatch.job_finished(w)
    ws = {w2["worker_id"]: w2 for w2 in dispatch.list_workers()}
    assert ws[w]["jobs_done"] == 1
    assert ws[w]["current_job_id"] is None


def test_reap_dead_worker_local():
    w = dispatch.register_worker("colab")
    dispatch.heartbeat(w, "job-stuck")
    # force the heartbeat stale
    c = dispatch._connect()
    c.execute(
        "UPDATE forge_workers SET last_heartbeat=? WHERE worker_id=?",
        (time.time() - 9999, w),
    )
    c.commit()
    rep = dispatch.reap_dead_workers()
    assert len(rep["reaped_workers"]) == 1
    assert rep["reaped_workers"][0]["worker_id"] == w
    # no Supabase creds in test env -> re-queue skipped, reported honestly
    assert rep["supabase_configured"] is False
    assert rep["reaped_workers"][0]["requeued"] is False
    ws = {x["worker_id"]: x for x in dispatch.list_workers()}
    assert ws[w]["alive"] is False


def test_reap_leaves_fresh_workers_alone():
    w = dispatch.register_worker("kaggle")
    dispatch.heartbeat(w, "job-live")
    rep = dispatch.reap_dead_workers()
    assert all(r["worker_id"] != w for r in rep["reaped_workers"])


def test_status_snapshot():
    w = dispatch.register_worker("kaggle")
    dispatch.heartbeat(w)
    snap = dispatch.status_snapshot()
    assert snap["alive_by_platform"].get("kaggle", 0) >= 1
    assert isinstance(snap["workers"], list)
    assert "queue" in snap
    assert isinstance(snap["queue_depth"], int)


def test_app_imports_with_dispatch():
    # The actual bug: main.py imported app.dispatch, which did not exist.
    import app.main  # noqa: F401

    assert app.main.app is not None
