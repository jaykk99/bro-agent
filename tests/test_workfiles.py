"""Tests for the work-files browser API (list/download/delete). No network."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-workfiles-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture()
def work(tmp_path, monkeypatch):
    root = str(tmp_path / "work")
    os.makedirs(os.path.join(root, "reports"))
    os.makedirs(os.path.join(root, "uploads"))
    with open(os.path.join(root, "notes.txt"), "w") as f:
        f.write("hello work files")
    with open(os.path.join(root, "reports", "r1.md"), "w") as f:
        f.write("# report")
    with open(os.path.join(root, "uploads", "up.bin"), "wb") as f:
        f.write(b"\x00\x01\x02")
    monkeypatch.setenv("BRO_WORKDIR", root)
    return root


def test_list_root(work):
    r = client.get("/api/files")
    assert r.status_code == 200
    d = r.json()
    assert d["path"] == ""
    names = [e["name"] for e in d["entries"]]
    # Directories sort first.
    assert names.index("reports") < names.index("notes.txt")
    by_name = {e["name"]: e for e in d["entries"]}
    assert by_name["reports"]["is_dir"] is True
    assert by_name["reports"]["bytes"] is None
    assert by_name["notes.txt"]["is_dir"] is False
    assert by_name["notes.txt"]["bytes"] == len("hello work files")
    assert by_name["notes.txt"]["path"] == "notes.txt"


def test_list_subdir(work):
    r = client.get("/api/files", params={"path": "reports"})
    assert r.status_code == 200
    assert [e["name"] for e in r.json()["entries"]] == ["r1.md"]


def test_list_missing_and_file(work):
    assert client.get("/api/files", params={"path": "nope"}).status_code == 404
    assert client.get("/api/files", params={"path": "notes.txt"}).status_code == 400


def test_download_exact_bytes(work):
    r = client.get("/api/files/download", params={"path": "notes.txt"})
    assert r.status_code == 200
    assert r.content == b"hello work files"
    assert r.headers["content-length"] == str(len("hello work files"))
    assert "notes.txt" in r.headers["content-disposition"]


def test_download_upload_via_workfiles_path(work):
    # The uploads table's download link goes through this same route.
    r = client.get("/api/files/download", params={"path": "uploads/up.bin"})
    assert r.status_code == 200
    assert r.content == b"\x00\x01\x02"


def test_download_rejects(work):
    assert client.get("/api/files/download").status_code == 400  # no path
    assert client.get("/api/files/download", params={"path": "nope.txt"}).status_code == 404
    assert client.get("/api/files/download", params={"path": "reports"}).status_code == 400  # dir
    assert client.get("/api/files/download", params={"path": "../secret"}).status_code == 400
    assert client.get("/api/files/download", params={"path": "/etc/passwd"}).status_code == 400


def test_list_rejects_escape(work):
    assert client.get("/api/files", params={"path": "../"}).status_code == 400
    assert client.get("/api/files", params={"path": "/etc"}).status_code == 400


def test_symlink_escape_blocked(work):
    outside = os.path.join(os.path.dirname(work), "outside")
    os.makedirs(outside)
    with open(os.path.join(outside, "secret.txt"), "w") as f:
        f.write("nope")
    os.symlink(outside, os.path.join(work, "link"))
    assert client.get("/api/files", params={"path": "link"}).status_code == 400
    assert client.get("/api/files/download", params={"path": "link/secret.txt"}).status_code == 400


def test_delete_file(work):
    r = client.delete("/api/files", params={"path": "notes.txt"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert not os.path.exists(os.path.join(work, "notes.txt"))
    assert client.delete("/api/files", params={"path": "notes.txt"}).status_code == 404


def test_delete_rejects(work):
    assert client.delete("/api/files").status_code == 400  # no path
    assert client.delete("/api/files", params={"path": "reports"}).status_code == 400  # dir
    assert client.delete("/api/files", params={"path": "../x"}).status_code == 400


def test_access_gate_blocks(work, monkeypatch):
    monkeypatch.setenv("BRO_ACCESS_CODE", "secret123")
    assert client.get("/api/files?access_code=wrong").status_code == 401
    assert client.get("/api/files?access_code=secret123").status_code == 200
    monkeypatch.delenv("BRO_ACCESS_CODE")
