"""Thread history pagination — ?limit=&offset= on /api/threads/{id}."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-page-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "page.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

from fastapi.testclient import TestClient  # noqa: E402

from app import memory  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def _seed(n: int = 10) -> str:
    tid = memory.create_thread("paged thread")
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        memory.add_message(tid, role, f"message {i}")
    return tid


def test_default_returns_all():
    tid = _seed()
    assert len(memory.get_thread_messages(tid)) == 10


def test_limit():
    tid = _seed()
    msgs = memory.get_thread_messages(tid, limit=4)
    assert len(msgs) == 4
    assert msgs[0]["content"] == "message 0"  # oldest-first window


def test_offset():
    tid = _seed()
    msgs = memory.get_thread_messages(tid, limit=4, offset=6)
    assert [m["content"] for m in msgs] == [f"message {i}" for i in range(6, 10)]


def test_offset_past_end_is_empty():
    tid = _seed(3)
    assert memory.get_thread_messages(tid, limit=5, offset=10) == []


def test_endpoint_pagination():
    tid = _seed()
    r = client.get(f"/api/threads/{tid}?limit=3&offset=2")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [m["content"] for m in msgs] == ["message 2", "message 3", "message 4"]


def test_endpoint_limit_clamped():
    tid = _seed(3)
    r = client.get(f"/api/threads/{tid}?limit=5000")
    assert r.status_code == 200
    assert len(r.json()["messages"]) == 3


def test_endpoint_missing_thread_404():
    assert client.get("/api/threads/nope?limit=5").status_code == 404
