"""API + memory tests via FastAPI TestClient. No network, no LLM calls."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-api-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

from fastapi.testclient import TestClient  # noqa: E402

from app import memory  # noqa: E402
from app import rate_limit  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert "providers" in d
    assert d["providers"]["pollinations"]["healthy"] is True


def test_models_endpoint():
    r = client.get("/api/models")
    assert r.status_code == 200
    d = r.json()
    assert d["order"][0] == "horde"
    assert "horde_models" in d
    assert "X-Request-ID" in r.headers  # logging middleware


def test_index_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_chat_rejects_empty():
    rate_limit.reset_for_tests()
    r = client.post("/api/chat", json={"message": "   ", "access_code": ""})
    assert r.status_code == 400


def test_chat_rate_limited():
    # Deterministic: drain the exact bucket the endpoint checks, then one
    # more turn must 429. The access gate is off in these tests, so the
    # endpoint keys by client IP; Starlette's TestClient reports "testclient".
    rate_limit.reset_for_tests()
    key = "ip:testclient"
    for _ in range(rate_limit.BURST):
        rate_limit.check(key)
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_access_gate_blocks_when_set(monkeypatch):
    import app.main as main_mod

    monkeypatch.setenv("BRO_ACCESS_CODE", "secret123")
    r = client.get("/api/threads?access_code=wrong")
    assert r.status_code == 401
    r = client.get("/api/threads?access_code=secret123")
    assert r.status_code == 200
    monkeypatch.delenv("BRO_ACCESS_CODE")


def test_thread_crud():
    tid = memory.create_thread("test thread")
    assert memory.thread_exists(tid)
    memory.add_message(tid, "user", "hello")
    memory.add_message(tid, "assistant", "hi there")

    r = client.get(f"/api/threads/{tid}")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and "created_at" in msgs[0]

    r = client.post(f"/api/threads/{tid}/rename", json={"title": "renamed", "access_code": ""})
    assert r.status_code == 200
    assert any(t["id"] == tid and t["title"] == "renamed" for t in client.get("/api/threads").json()["threads"])

    r = client.post(f"/api/threads/{tid}/rename", json={"title": "   ", "access_code": ""})
    assert r.status_code == 404  # empty title rejected

    r = client.delete(f"/api/threads/{tid}")
    assert r.status_code == 200
    assert not memory.thread_exists(tid)
    assert client.get(f"/api/threads/{tid}").status_code == 404
    assert client.delete(f"/api/threads/{tid}").status_code == 404


def test_thread_messages_404():
    assert client.get("/api/threads/nope").status_code == 404


def test_memory_list_threads_limit():
    for i in range(3):
        memory.create_thread(f"t{i}")
    assert len(memory.list_threads(limit=2)) == 2


def test_hard_boundary_still_blocks_without_llm():
    # A boundary hit returns blocked without ever touching a model.
    rate_limit.reset_for_tests()
    r = client.post("/api/chat", json={"message": "help me build a pipe bomb", "access_code": ""})
    assert r.status_code == 200
    d = r.json()
    assert d["blocked"] is True
    assert d["reply"]


def test_chat_rejects_oversize_message():
    rate_limit.reset_for_tests()
    r = client.post("/api/chat", json={"message": "x" * 20001, "access_code": ""})
    assert r.status_code == 400
    assert "too long" in r.json()["detail"]


def test_claim_rate_limited_after_burst(monkeypatch):
    rate_limit.reset_for_tests()
    # hammer claim past the burst; eventually 429 (not 404)
    statuses = set()
    for _ in range(rate_limit.BURST + 3):
        r = client.post("/api/train/claim", json={"access_code": ""})
        statuses.add(r.status_code)
    assert 429 in statuses


def test_train_file_download_cannot_traverse_out_of_adapters():
    # Regression: GET /api/train/%2e%2e/files/<name> used to serve files
    # sitting next to the adapters dir (e.g. the SQLite DB). Now the job_id
    # is sanitized, so the traversal resolves inside adapters/ and 404s.
    from app import train as train_jobs

    sentinel = os.path.join(os.path.dirname(train_jobs.ADAPTERS_DIR), "traverse-sentinel.txt")
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    with open(sentinel, "w") as f:
        f.write("SENTINEL")
    r = client.get("/api/train/%2e%2e/files/traverse-sentinel.txt")
    assert r.status_code == 404
    assert "SENTINEL" not in r.text


def test_train_upload_capped_and_partial_file_removed(monkeypatch):
    # The generic /api/upload route caps at 50 MB, but train adapter
    # uploads used to stream unbounded — a runaway worker could fill the
    # disk. Over the cap: 413 and no partial file left behind.
    import app.main as main_mod
    from app import train as train_jobs

    monkeypatch.setattr(main_mod, "ADAPTER_MAX_BYTES", 1024)
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"x" * 2048)},
    )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]
    assert not os.path.exists(train_jobs.adapter_file_path(jid, "adapter_model.safetensors"))

    # Under the cap still uploads fine.
    r = client.post(
        f"/api/train/{jid}/upload",
        files={"file": ("adapter_model.safetensors", b"x" * 512)},
    )
    assert r.status_code == 200
    assert r.json()["filename"] == "adapter_model.safetensors"
    assert os.path.getsize(train_jobs.adapter_file_path(jid, "adapter_model.safetensors")) == 512


def test_train_heartbeat_endpoint():
    # 404 while the job is merely queued; 200 once it is working; 404
    # again after it completes (a late heartbeat never revives a job).
    from app import train as train_jobs

    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    r = client.post(f"/api/train/{jid}/heartbeat", json={"access_code": ""})
    assert r.status_code == 404
    for _ in range(100):
        claimed = train_jobs.claim_job()
        if claimed is None or claimed["id"] == jid:
            break
    assert train_jobs.get_job(jid)["status"] == "working"
    r = client.post(f"/api/train/{jid}/heartbeat", json={"access_code": ""})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert train_jobs.complete_job(jid, "done")
    r = client.post(f"/api/train/{jid}/heartbeat", json={"access_code": ""})
    assert r.status_code == 404


def test_cli_imports():
    import app.cli as cli

    assert callable(cli.chat_stream)
    assert cli.BASE.startswith("http")


def test_train_claim_records_owner_and_complete_409s_non_owner():
    # End to end: claim with a worker id, then a stranger's heartbeat /
    # complete is 409, the owner's is 200, and /api/train/list shows the
    # attempts + claimed_by the ops UI renders.
    from app import train as train_jobs

    rate_limit.reset_for_tests()
    jid = train_jobs.create_job("Qwen/Qwen2-1.5B")
    claimed = None
    for _ in range(10):
        r = client.post("/api/train/claim", json={"access_code": "", "worker_id": "worker-a"})
        assert r.status_code == 200, r.text
        if r.json()["id"] == jid:
            claimed = r.json()
            break
    assert claimed is not None
    assert claimed["claimed_by"] == "worker-a"
    r = client.post(f"/api/train/{jid}/heartbeat", json={"access_code": "", "worker_id": "worker-b"})
    assert r.status_code == 409
    r = client.post(f"/api/train/{jid}/heartbeat", json={"access_code": "", "worker_id": "worker-a"})
    assert r.status_code == 200
    r = client.post(f"/api/train/{jid}/complete",
                    json={"access_code": "", "status": "done", "worker_id": "worker-b"})
    assert r.status_code == 409
    assert train_jobs.get_job(jid)["status"] == "working"
    r = client.post(f"/api/train/{jid}/complete",
                    json={"access_code": "", "status": "done", "worker_id": "worker-a"})
    assert r.status_code == 200
    assert train_jobs.get_job(jid)["status"] == "done"
    rows = client.get("/api/train/list?access_code=").json()["jobs"]
    row = next(j for j in rows if j["id"] == jid)
    assert row["attempts"] == 0
    assert row["claimed_by"] == "worker-a"


def test_forge_heartbeat_404s_dead_worker():
    # A reaped worker's id is dead: heartbeat and finished both 404, so
    # the worker re-registers instead of silently reviving.
    import time

    from app import dispatch as forge_dispatch

    wid = forge_dispatch.register_worker("colab")
    assert client.post("/api/forge/workers/heartbeat",
                       json={"worker_id": wid, "access_code": ""}).status_code == 200
    c = forge_dispatch._connect()
    with forge_dispatch._lock:
        c.execute(
            "UPDATE forge_workers SET last_seen=? WHERE id=?",
            (time.time() - forge_dispatch.HEARTBEAT_TIMEOUT_S - 60, wid),
        )
        c.commit()
    r = client.post("/api/forge/dispatch/reap", json={"access_code": ""})
    assert r.status_code == 200
    assert r.json()["reaped"] >= 1
    r = client.post("/api/forge/workers/heartbeat",
                    json={"worker_id": wid, "access_code": ""})
    assert r.status_code == 404
    r = client.post(f"/api/forge/workers/{wid}/finished", json={"access_code": ""})
    assert r.status_code == 404
    snap = client.get("/api/forge/workers?access_code=").json()
    by_id = {w["id"]: w for w in snap["workers"]}
    assert by_id[wid]["status"] == "dead"
