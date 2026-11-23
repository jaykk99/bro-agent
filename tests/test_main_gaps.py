"""Pin the remaining untested paths in app/main.py.

Coverage gaps left after test_api.py / test_chat.py / test_train.py:
request-log middleware's re-raise path, /api/models' Horde fallback,
thread export's short-page break, the train heartbeat 404 that only
fires when the job IS working, the adapter-cap env fallbacks, the
work-file download/delete ValueError branches, the generic /api/upload
413, the chat-stream producer crash + consumer-disconnect cancel, the
train create/get/complete/upload endpoints, the upload-route symlink
re-checks and commit-time 413, and the forge register/finished
endpoints. No network, no LLM: fakes and planted symlinks only.
"""

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-main-gaps-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)

from fastapi.testclient import TestClient  # noqa: E402

import app.agent as agent_mod  # noqa: E402
import app.main as main_mod  # noqa: E402
from app import horde_llm as horde_mod  # noqa: E402
from app import memory, rate_limit, train as train_jobs, workfiles  # noqa: E402
from app.router import get_router  # noqa: E402

client = TestClient(main_mod.app)


def _fresh_router():
    r = get_router()
    for p in ("pollinations", "horde"):
        r.record_success(p)
    rate_limit.reset_for_tests()


# ---- middleware re-raise (lines 127-128) ----


def test_middleware_reraises_handler_errors(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(memory, "list_threads", boom)
    with pytest.raises(RuntimeError, match="db down"):
        client.get("/api/threads")


# ---- /api/models Horde fallback (lines 211-212) ----


def test_models_endpoint_survives_horde_list_failure(monkeypatch):
    def boom(_configured):
        raise RuntimeError("langchain missing")

    monkeypatch.setattr(horde_mod, "resolve_models", boom)
    r = client.get("/api/models")
    assert r.status_code == 200
    assert r.json()["horde_models"] == []


# ---- thread export short-page break (line 248) ----


def test_thread_export_small_thread():
    tid = memory.create_thread("export me")
    memory.add_message(tid, "user", "hello")
    memory.add_message(tid, "assistant", "hi back")
    r = client.get(f"/api/threads/{tid}/export")
    assert r.status_code == 200
    assert "hello" in r.text and "hi back" in r.text
    assert "bro-chat-" in r.headers["Content-Disposition"]


def test_thread_export_empty_thread():
    # No messages at all: the first page is empty, so the loop breaks
    # on the empty-page branch instead of the short-page branch.
    tid = memory.create_thread("empty export")
    r = client.get(f"/api/threads/{tid}/export")
    assert r.status_code == 200
    assert "# bro chat export" in r.text


# ---- train heartbeat 404 when the job IS working (line 298) ----
# _check_job_owner raises its own 404 for a non-working job (covered in
# test_api.py); this line only fires when the job is working but the
# heartbeat UPDATE finds nothing (a race the route must still 404 on).


def test_train_heartbeat_404_when_update_finds_nothing(monkeypatch):
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    for _ in range(10):
        claimed = train_jobs.claim_job("w1")
        if claimed is None or claimed["id"] == jid:
            break
    assert train_jobs.get_job(jid)["status"] == "working"
    monkeypatch.setattr(train_jobs, "heartbeat_job", lambda *a, **k: False)
    r = client.post(f"/api/train/{jid}/heartbeat", json={"access_code": "", "worker_id": "w1"})
    assert r.status_code == 404


# ---- adapter cap env fallbacks (lines 313-314, 332-333, 349-350) ----


def test_adapter_max_bytes_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("BRO_ADAPTER_MAX_BYTES", "not-a-number")
    assert main_mod._adapter_max_bytes() == 2 * 1024**3


def test_adapter_job_max_bytes_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("BRO_ADAPTER_JOB_MAX_BYTES", "not-a-number")
    assert main_mod._adapter_job_max_bytes() == max(4 * 1024**3, main_mod.ADAPTER_MAX_BYTES)


def test_adapter_total_max_bytes_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("BRO_ADAPTER_TOTAL_MAX_BYTES", "not-a-number")
    assert main_mod._adapter_total_max_bytes() == max(
        20 * 1024**3, main_mod._adapter_job_max_bytes()
    )


# ---- work-file download/delete ValueError branches (lines 436-437, 482-483) ----


def test_work_file_download_not_a_regular_file(monkeypatch):
    def boom(root, rel):
        raise ValueError("not a regular file")

    monkeypatch.setattr(workfiles, "open_work_file", boom)
    r = client.get("/api/files/download?path=x.bin")
    assert r.status_code == 400
    assert "not a regular file" in r.json()["detail"]


def test_work_file_delete_not_a_regular_file(monkeypatch):
    def boom(root, rel):
        raise ValueError("not a regular file")

    monkeypatch.setattr(workfiles, "delete_work_file", boom)
    r = client.delete("/api/files?path=x.bin")
    assert r.status_code == 400
    assert "not a regular file" in r.json()["detail"]


def test_upload_delete_404():
    assert client.delete("/api/uploads/does-not-exist.bin").status_code == 404


# ---- generic /api/upload 413 (lines 520-522) ----


def test_generic_upload_413_removes_partial(monkeypatch):
    monkeypatch.setattr(main_mod, "UPLOAD_MAX_BYTES", 10)
    r = client.post("/api/upload", files={"file": ("big.bin", b"x" * 20)})
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]
    # The partial file is gone from the workdir.
    from app.sandbox import workdir

    assert not os.path.exists(os.path.join(workdir(), "uploads", "big.bin"))


# ---- chat-stream producer crash (lines 700-702) ----


def test_chat_stream_producer_crash_yields_error_event(monkeypatch):
    _fresh_router()

    def boom():
        raise RuntimeError("router down")

    monkeypatch.setattr(main_mod, "get_router", boom)
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    assert r.status_code == 200
    evts = [json.loads(l[5:]) for l in r.text.split("\n") if l.strip().startswith("data:")]
    err = next(e for e in evts if e["type"] == "error")
    assert "internal error: router down" in err["error"]


# ---- chat-stream consumer disconnect cancels the turn (line 717) ----
# TestClient streaming never tells the server the client left (the
# generator keeps pinging into the void), so drive the endpoint's SSE
# generator directly and close it mid-turn: the finally must cancel
# the still-pending turn task instead of leaving it queued forever.


def test_chat_stream_disconnect_cancels_pending_turn(monkeypatch):
    import json as _json

    from starlette.requests import Request

    _fresh_router()

    async def hang(preferred=None):
        await asyncio.sleep(3600)
        return None, "horde", False

    monkeypatch.setattr(agent_mod, "get_cached_agent", hang)

    async def _drive():
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
            "client": ("testclient", 50000),
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
        }
        req = main_mod.ChatRequest(message="hi", access_code="")
        resp = await main_mod.chat(req, Request(scope))
        it = resp.body_iterator
        first = await it.__anext__()
        assert "thread" in first  # turn started; close without reading to done
        await it.aclose()
        await asyncio.sleep(0.1)  # let the cancellation propagate
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert pending == []

    asyncio.run(_drive())


# ---- train endpoints (lines 727-732, 743-747, 769, 773, 802) ----


def test_train_create_and_get_round_trip():
    r = client.post("/api/train", json={"model_repo": "Qwen/Qwen2-1.5B", "access_code": ""})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    r = client.get(f"/api/train/{jid}")
    assert r.status_code == 200
    assert r.json()["model_repo"] == "Qwen/Qwen2-1.5B"


def test_train_create_rejects_empty_repo():
    r = client.post("/api/train", json={"model_repo": "   ", "access_code": ""})
    assert r.status_code == 400
    assert "model_repo is required" in r.json()["detail"]


def test_train_get_404():
    assert client.get("/api/train/no-such-job").status_code == 404


def test_train_claim_404_when_queue_empty(monkeypatch):
    monkeypatch.setattr(train_jobs, "claim_job", lambda *a, **k: None)
    r = client.post("/api/train/claim", json={"access_code": ""})
    assert r.status_code == 404
    assert "no queued jobs" in r.json()["detail"]


def test_train_complete_rejects_bad_status():
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    r = client.post(
        f"/api/train/{jid}/complete",
        json={"access_code": "", "status": "weird"},
    )
    assert r.status_code == 400
    assert "status must be done or failed" in r.json()["detail"]


def test_train_complete_404_when_not_working():
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")  # queued, never claimed
    r = client.post(
        f"/api/train/{jid}/complete",
        json={"access_code": "", "status": "done"},
    )
    assert r.status_code == 404


def test_train_complete_404_when_update_finds_nothing(monkeypatch):
    # _check_job_owner raises its own 404 for a non-working job; this
    # line only fires when the job IS working but the complete UPDATE
    # finds nothing (a race the route must still 404 on).
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    for _ in range(10):
        claimed = train_jobs.claim_job("w1")
        if claimed is None or claimed["id"] == jid:
            break
    assert train_jobs.get_job(jid)["status"] == "working"
    monkeypatch.setattr(train_jobs, "complete_job", lambda *a, **k: False)
    r = client.post(
        f"/api/train/{jid}/complete",
        json={"access_code": "", "status": "done", "worker_id": "w1"},
    )
    assert r.status_code == 404


def test_train_upload_404_for_unknown_job():
    r = client.post(
        "/api/train/no-such-job/upload",
        files={"file": ("adapter_model.safetensors", b"x")},
    )
    assert r.status_code == 404


def test_train_upload_rejects_bad_sha256_format():
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"x")},
        data={"sha256": "not-hex"},
    )
    assert r.status_code == 400
    assert "sha256 must be 64 lowercase hex" in r.json()["detail"]


def test_train_upload_honours_sha256_match():
    # A matching worker-supplied hash lands the file (happy path for the
    # integrity check whose mismatch branch is already covered).
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    payload = b"adapter-bytes"
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", payload)},
        data={"sha256": hashlib.sha256(payload).hexdigest()},
    )
    assert r.status_code == 200
    assert r.json()["sha256"] == hashlib.sha256(payload).hexdigest()


# ---- upload-route symlink refusals (lines 880-881, 891) ----


def test_train_upload_refuses_symlinked_job_dir():
    # The link must point INSIDE the adapters tree: adapter_file_path's
    # own realpath check refuses an outward link first (a different
    # 400), so an inward link is refused by the pre-open symlink check.
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    other = train_jobs.create_job("Qwen/Qwen2-1.5B")
    target_dir = os.path.dirname(train_jobs.adapter_file_path(other, "x", create_dir=True))
    d = train_jobs.adapter_file_path(jid, "x", create_dir=True)
    job_dir = os.path.dirname(d)
    os.rmdir(job_dir)
    os.symlink(target_dir, job_dir)
    try:
        r = client.post(
            f"/api/train/{jid}/upload",
            files={"file": ("adapter_model.safetensors", b"x")},
        )
        assert r.status_code == 400
        assert "symlink" in r.json()["detail"]
    finally:
        os.unlink(job_dir)


def test_train_upload_pinned_open_failure_maps_to_400(monkeypatch):
    # The pinned open_adapter_job_dir can only fail after the pre-checks
    # when the dir is swapped mid-request (a race): fault-inject it to
    # pin the route's FileNotFoundError -> 400 mapping.
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")

    def boom(job_id):
        raise FileNotFoundError("adapter job dir not found")

    monkeypatch.setattr(train_jobs, "open_adapter_job_dir", boom)
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"x")},
    )
    assert r.status_code == 400
    assert "symlink" in r.json()["detail"]


def test_train_upload_dest_stat_symlink_maps_to_400(monkeypatch, tmp_path):
    # Same race shape for the destination stat: the pre-open check
    # passed, then the entry became a symlink before the stat. Inject
    # a symlink stat on the first dir-fd stat call.
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    target = tmp_path / "outside"
    target.write_bytes(b"nope")
    link = tmp_path / "planted"
    os.symlink(str(target), str(link))
    link_st = os.lstat(str(link))

    real_stat = os.stat
    calls = {"n": 0}

    def counting_stat(path, *a, **k):
        if k.get("dir_fd") is not None and path == "adapter_model.safetensors":
            calls["n"] += 1
            if calls["n"] == 1:
                return link_st
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "stat", counting_stat)
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"x")},
    )
    assert r.status_code == 400
    assert "symlink" in r.json()["detail"]


def test_train_upload_refuses_symlinked_dest_file():
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    dest = train_jobs.adapter_file_path(jid, "adapter_model.safetensors", create_dir=True)
    os.symlink("/etc/passwd", dest)
    try:
        r = client.post(
            f"/api/train/{jid}/upload",
            files={"file": ("adapter_model.safetensors", b"x")},
        )
        assert r.status_code == 400
        assert "symlink" in r.json()["detail"]
    finally:
        os.unlink(dest)


# ---- upload cleanup: unlink failure is swallowed (lines 943-944) ----


def test_train_upload_cleanup_survives_unlink_failure(monkeypatch):
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    real_unlink = os.unlink

    def flaky_unlink(path, *a, **k):
        if isinstance(path, str) and path.startswith(".upload-"):
            raise OSError("unlink down")
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(os, "unlink", flaky_unlink)
    # Wrong hash -> 422; the temp-file cleanup unlink fails and is swallowed.
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"bytes")},
        data={"sha256": "0" * 64},
    )
    assert r.status_code == 422
    assert "sha256 mismatch" in r.json()["detail"]


# ---- commit-time re-check: dest swapped to a symlink (lines 965-969) ----


def test_train_upload_commit_recheck_refuses_swapped_symlink(monkeypatch, tmp_path):
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    target = tmp_path / "outside"
    target.write_bytes(b"nope")
    link = tmp_path / "planted"
    os.symlink(str(target), str(link))
    link_st = os.lstat(str(link))

    real_stat = os.stat
    calls = {"n": 0}

    def counting_stat(path, *a, **k):
        if k.get("dir_fd") is not None and path == "adapter_model.safetensors":
            calls["n"] += 1
            if calls["n"] == 2:
                return link_st  # swapped in between the two checks
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "stat", counting_stat)
    real_unlink = os.unlink

    def flaky_unlink(path, *a, **k):
        if isinstance(path, str) and path.startswith(".upload-"):
            raise OSError("unlink down")
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(os, "unlink", flaky_unlink)
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"bytes")},
    )
    assert r.status_code == 400
    assert "symlink" in r.json()["detail"]
    assert calls["n"] == 2
    assert not os.path.exists(train_jobs.adapter_file_path(jid, "adapter_model.safetensors"))


# ---- commit-time re-check: total cap trips late (lines 972-976) ----


def test_train_upload_commit_recheck_413_when_total_grows(monkeypatch):
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    monkeypatch.setenv("BRO_ADAPTER_TOTAL_MAX_BYTES", "1024")
    real_total = train_jobs.adapters_total_bytes
    calls = {"n": 0}

    def growing_total():
        calls["n"] += 1
        return 0 if calls["n"] == 1 else 10**12  # another job landed mid-stream

    monkeypatch.setattr(train_jobs, "adapters_total_bytes", growing_total)
    real_unlink = os.unlink

    def flaky_unlink(path, *a, **k):
        if isinstance(path, str) and path.startswith(".upload-"):
            raise OSError("unlink down")
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(os, "unlink", flaky_unlink)
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"x" * 100)},
    )
    assert r.status_code == 413
    assert calls["n"] == 2
    assert not os.path.exists(train_jobs.adapter_file_path(jid, "adapter_model.safetensors"))


# ---- forge endpoints (lines 1061-1063, 1080) ----


def test_forge_register_and_finished_round_trip():
    r = client.post(
        "/api/forge/workers/register",
        json={"platform": "colab", "access_code": ""},
    )
    assert r.status_code == 200
    wid = r.json()["worker_id"]
    r = client.post(f"/api/forge/workers/{wid}/finished", json={"access_code": ""})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ---- module entry point (lines 1098-1100) ----
# The subprocess runs under coverage too, so the __main__ guard lines
# are measured, not just the server startup: SIGINT gives uvicorn a
# clean shutdown, which lets coverage write its data file.


def test_main_entrypoint_starts_uvicorn(tmp_path):
    import signal
    import time

    cov_file = tmp_path / ".coverage-entry"
    env = dict(os.environ)
    env["PORT"] = "18131"
    env["BRO_DB_PATH"] = str(tmp_path / "entry.db")
    env["BRO_REAP_INTERVAL_S"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--data-file",
            str(cov_file),
            "-m",
            "app.main",
        ],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        started = False
        deadline = time.time() + 15
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            line = proc.stdout.readline()
            if "Uvicorn running on" in line:
                started = True
                break
        assert started, "server did not start"
        assert proc.poll() is None, "server exited early"
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    from coverage import Coverage

    cov = Coverage(data_file=str(cov_file))
    cov.load()
    _filename, _stmts, _excluded, missing, _misf = cov.analysis2(
        os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
    )
    assert 1098 not in missing and 1099 not in missing and 1100 not in missing
