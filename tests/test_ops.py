"""Tests for train retry/stuck-failover, usage stats, export, upload."""

import io
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-ops-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["BRO_WORKDIR"] = os.path.join(_tmp, "work")
os.environ.pop("BRO_ACCESS_CODE", None)

from fastapi.testclient import TestClient  # noqa: E402

from app import stats, train  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def _drain_queue():
    while train.claim_job():
        pass


def test_retry_failed_job():
    _drain_queue()
    jid = train.create_job("m")
    claimed = train.claim_job()
    assert claimed["id"] == jid
    assert train.complete_job(jid, "failed", error="boom") is True
    assert train.get_job(jid)["status"] == "failed"

    r = client.post(f"/api/train/{jid}/retry", json={"access_code": ""})
    assert r.status_code == 200
    job = train.get_job(jid)
    assert job["status"] == "queued"
    assert job["attempts"] == 1


def test_retry_caps_attempts():
    _drain_queue()
    jid = train.create_job("m2")
    for _ in range(3):
        train.claim_job()
        train.complete_job(jid, "failed", error="x")
        assert train.retry_job(jid) is True
    # 4th failure: out of attempts
    train.claim_job()
    train.complete_job(jid, "failed", error="x")
    assert train.retry_job(jid) is False
    r = client.post(f"/api/train/{jid}/retry", json={"access_code": ""})
    assert r.status_code == 404


def test_retry_refuses_non_failed():
    _drain_queue()
    jid = train.create_job("m3")  # still queued
    assert train.retry_job(jid) is False
    assert train.retry_job("nope") is False


def test_fail_stuck_jobs():
    _drain_queue()
    jid = train.create_job("m4")
    train.claim_job()  # -> working
    # backdate updated_at so it counts as stuck
    c = train._connect()
    with train._lock:
        c.execute("UPDATE train_jobs SET updated_at=? WHERE id=?", (time.time() - 7200, jid))
        c.commit()
    r = client.post("/api/train/reap-stuck", json={"access_code": ""})
    assert r.status_code == 200
    assert jid in r.json()["requeued"]
    job = train.get_job(jid)
    assert job["status"] == "queued"
    assert job["attempts"] == 1
    # fresh working jobs are not touched
    jid2 = train.create_job("m5")
    train.claim_job()
    assert train.fail_stuck_jobs(max_working_s=3600) == []


def test_stats_record_and_get():
    stats.record_turn("pollinations", 100, 50, day="2026-10-20")
    stats.record_turn("pollinations", 200, 60, day="2026-10-20")
    stats.record_turn("horde", 10, 5, day="2026-10-20")
    rows = stats.get_stats(days=30)
    today = [r for r in rows if r["day"] == "2026-10-20"]
    p = next(r for r in today if r["provider"] == "pollinations")
    assert p["turns"] == 2 and p["in_chars"] == 300 and p["out_chars"] == 110
    r = client.get("/api/stats")
    assert r.status_code == 200
    assert any(s["provider"] == "horde" for s in r.json()["stats"])


def test_thread_export():
    from app import memory

    tid = memory.create_thread("export me")
    memory.add_message(tid, "user", "hello")
    memory.add_message(tid, "assistant", "hi")
    r = client.get(f"/api/threads/{tid}/export")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert "## Jay" in r.text and "## bro" in r.text
    assert "hello" in r.text
    assert client.get("/api/threads/nope/export").status_code == 404


def test_thread_export_past_1000_messages():
    """Export must include the whole thread, not just the first 1000 messages.

    Regression: the export route used to fetch a single page (limit=1000),
    so a long thread's export silently dropped its most recent messages.
    """
    from app import memory

    tid = memory.create_thread("long export")
    for i in range(1005):
        role = "user" if i % 2 == 0 else "assistant"
        memory.add_message(tid, role, f"export message {i}")
    r = client.get(f"/api/threads/{tid}/export")
    assert r.status_code == 200
    assert "export message 0" in r.text
    assert "export message 1004" in r.text  # past the old 1000-message cap
    assert r.text.count("\n## ") == 1005


def test_upload():
    content = b"x" * 100
    r = client.post(
        "/api/upload",
        files={"file": ("../../../evil.txt", io.BytesIO(content))},
        data={"access_code": ""},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["bytes"] == 100
    # traversal neutralized: flat filename inside uploads/, sandbox-contained
    assert d["path"].startswith("uploads" + os.sep)
    assert "/" not in d["path"][len("uploads" + os.sep):]
    assert not os.path.basename(d["path"]).startswith(".")
    full = os.path.join(os.environ["BRO_WORKDIR"], d["path"])
    assert os.path.isfile(full)
