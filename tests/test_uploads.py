"""Upload management tests — list + delete via /api/uploads."""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-uploads-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "uploads.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    # Per-test workdir via monkeypatch (not module-level os.environ):
    # pytest imports every test module before running any, so a
    # module-level BRO_WORKDIR here would leak into other modules' tests
    # (e.g. test_ops.test_upload, which runs first).
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("BRO_WORKDIR", str(work))
    return work


def _upload(name: str, content: bytes = b"hello"):
    return client.post(
        "/api/upload",
        files={"file": (name, io.BytesIO(content), "application/octet-stream")},
        data={"access_code": ""},
    )


def test_upload_list_empty():
    r = client.get("/api/uploads")
    assert r.status_code == 200
    assert r.json()["uploads"] == []


def test_upload_then_list_then_delete():
    r = _upload("notes.txt")
    assert r.status_code == 200

    r = client.get("/api/uploads")
    files = r.json()["uploads"]
    assert len(files) == 1
    assert files[0]["name"] == "notes.txt"
    assert files[0]["bytes"] == 5
    assert files[0]["mtime"] > 0

    r = client.delete("/api/uploads/notes.txt")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.get("/api/uploads")
    assert r.json()["uploads"] == []


def test_delete_missing_is_404():
    r = client.delete("/api/uploads/no-such-file.txt")
    assert r.status_code == 404


def test_delete_path_traversal_is_sanitized_not_escaped(_workdir):
    # A traversal-style name gets sanitized to a single safe segment
    # (slashes -> underscores), so it can never point outside uploads/.
    # The delete route applies the same sanitizer, so the raw traversal
    # string and the listed name resolve to the same flat file.
    r = _upload("../../evil.txt")
    assert r.status_code == 200
    listed = [f["name"] for f in client.get("/api/uploads").json()["uploads"]]
    assert listed == ["_.._evil.txt"]  # one flat segment, no slashes
    work = os.environ["BRO_WORKDIR"]
    assert not os.path.exists(os.path.join(work, "evil.txt"))
    assert not os.path.exists(os.path.join(os.path.dirname(work), "evil.txt"))
    r = client.delete("/api/uploads/_.._evil.txt")
    assert r.status_code == 200
    assert client.get("/api/uploads").json()["uploads"] == []
