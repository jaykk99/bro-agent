"""Tests for the automatic stuck-job reaper loop (no network needed)."""

import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-reaper-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

from app import dispatch  # noqa: E402
from app import reaper  # noqa: E402
from app import train  # noqa: E402


def _backdate_job(jid: str, secs_ago: float) -> None:
    c = train._connect()
    with train._lock:
        c.execute(
            "UPDATE train_jobs SET updated_at=? WHERE id=?",
            (time.time() - secs_ago, jid),
        )
        c.commit()


def _backdate_worker(wid: str, secs_ago: float) -> None:
    c = dispatch._connect()
    with dispatch._lock:
        c.execute(
            "UPDATE forge_workers SET last_seen=? WHERE id=?",
            (time.time() - secs_ago, wid),
        )
        c.commit()


def _claim_job(jid: str) -> dict:
    for _ in range(100):
        c = train.claim_job()
        if c is None:
            break
        if c["id"] == jid:
            return c
    raise AssertionError(f"job {jid} was never claimed")


def test_run_reap_once_requeues_stuck_job_without_worker():
    # A job claimed directly via /api/train/claim has no worker row at
    # all — the forge reaper can't see it, only fail_stuck_jobs can.
    jid = train.create_job("model-reaper-a")
    _claim_job(jid)
    assert train.get_job(jid)["status"] == "working"
    _backdate_job(jid, 3600 + 60)
    out = reaper.run_reap_once()
    assert jid in out["requeued"]
    job = train.get_job(jid)
    assert job["status"] == "queued"
    assert job["attempts"] == 1


def test_run_reap_once_reaps_dead_worker_and_its_job():
    jid = train.create_job("model-reaper-b")
    _claim_job(jid)
    wid = dispatch.register_worker("colab")
    assert dispatch.heartbeat(wid, jid) is True
    _backdate_worker(wid, dispatch.HEARTBEAT_TIMEOUT_S + 60)
    out = reaper.run_reap_once()
    assert out["reaped_workers"] >= 1
    assert jid in out["requeued"]
    assert train.get_job(jid)["status"] == "queued"


def test_run_reap_once_retires_poison_job():
    jid = train.create_job("model-reaper-poison")
    _claim_job(jid)
    c = train._connect()
    with train._lock:
        c.execute(
            "UPDATE train_jobs SET attempts=? WHERE id=?",
            (train.MAX_ATTEMPTS, jid),
        )
        c.commit()
    _backdate_job(jid, 3600 + 60)
    out = reaper.run_reap_once()
    assert jid in out["failed"]
    assert jid not in out["requeued"]
    job = train.get_job(jid)
    assert job["status"] == "failed"
    assert "needs Jay" in (job["error"] or "")


def test_run_reap_once_survives_one_half_failing(monkeypatch):
    # A DB blip on the train half must not skip the forge half, and the
    # failure must be recorded in status(), not raised.
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(train, "fail_stuck_jobs", _boom)
    jid = train.create_job("model-reaper-c")
    _claim_job(jid)
    wid = dispatch.register_worker("kaggle")
    assert dispatch.heartbeat(wid, jid) is True
    _backdate_worker(wid, dispatch.HEARTBEAT_TIMEOUT_S + 60)
    before = reaper.status()["runs"]
    out = reaper.run_reap_once()
    assert out["reaped_workers"] >= 1
    assert jid in out["requeued"]
    st = reaper.status()
    assert st["runs"] == before + 1
    assert st["last_error"] and "train reap failed" in st["last_error"]
    assert st["last_result"] == out


def test_run_reap_once_records_status():
    before = reaper.status()["runs"]
    out = reaper.run_reap_once()
    st = reaper.status()
    assert st["runs"] == before + 1
    assert st["last_run_at"] is not None
    assert st["last_run_at"] <= time.time()
    assert st["last_result"] == out
    assert st["last_error"] is None
    assert set(out) == {"requeued", "failed", "reaped_workers"}


def test_reap_interval_parsing(monkeypatch):
    monkeypatch.delenv("BRO_REAP_INTERVAL_S", raising=False)
    assert reaper.reap_interval_s() == 600
    monkeypatch.setenv("BRO_REAP_INTERVAL_S", "0")
    assert reaper.reap_interval_s() == 0
    monkeypatch.setenv("BRO_REAP_INTERVAL_S", "30")
    assert reaper.reap_interval_s() == 60  # clamped up to the minimum
    monkeypatch.setenv("BRO_REAP_INTERVAL_S", "900")
    assert reaper.reap_interval_s() == 900
    monkeypatch.setenv("BRO_REAP_INTERVAL_S", "garbage")
    assert reaper.reap_interval_s() == 600


def test_background_loop_runs_and_stops():
    async def _body():
        assert await reaper.stop() is None  # no-op when not running
        assert reaper.start(interval_s=0.05) is True
        assert reaper.start(interval_s=0.05) is False  # already running
        before = reaper.status()["runs"]
        await asyncio.sleep(0.3)
        st = reaper.status()
        assert st["running"] is True
        assert st["runs"] >= before + 2  # immediate pass + repeats
        await reaper.stop()
        assert reaper.status()["running"] is False
        # Restartable after a stop.
        assert reaper.start(interval_s=0.05) is True
        await reaper.stop()
        assert reaper.status()["running"] is False

    asyncio.run(_body())


def test_start_disabled_when_interval_zero(monkeypatch):
    async def _body():
        monkeypatch.setenv("BRO_REAP_INTERVAL_S", "0")
        assert reaper.start() is False
        assert reaper.status()["enabled"] is False
        assert reaper.status()["running"] is False

    asyncio.run(_body())


def test_forge_workers_endpoint_reports_reaper(monkeypatch):
    # Lifespan wiring: entering the TestClient starts the loop, exiting
    # stops it — and the dispatcher snapshot carries the reaper state.
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setenv("BRO_REAP_INTERVAL_S", "60")
    before = reaper.status()["runs"]
    with TestClient(main.app) as client:
        assert reaper.status()["running"] is True
        # The loop's first pass runs in a thread on start; wait for it
        # instead of assuming it beat this GET (no fixed-sleep race).
        deadline = time.time() + 5
        while reaper.status()["runs"] < before + 1 and time.time() < deadline:
            time.sleep(0.05)
        assert reaper.status()["runs"] >= before + 1  # immediate first pass
        resp = client.get("/api/forge/workers")
        assert resp.status_code == 200
        body = resp.json()
        assert "reaper" in body
        assert body["reaper"]["enabled"] is True
        assert body["reaper"]["interval_s"] == 60
        assert body["reaper"]["runs"] >= before + 1
    assert reaper.status()["running"] is False


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("ALL REAPER TESTS PASS")
