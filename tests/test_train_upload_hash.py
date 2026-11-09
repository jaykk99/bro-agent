"""Train adapter upload integrity — sha256 verification + recorded hashes.

Proven gap (2026-11-09, before the fix): the upload route stored whatever
bytes arrived with no check at all, so a truncated or corrupted upload
(flaky Colab connection, worker-side partial file) landed as a good
adapter and only failed when Jay tried to load it. The route now hashes
the streamed bytes itself; a worker-supplied sha256 that mismatches is
a 422 and the file is NOT stored (the old adapter at that name survives,
like a 413). The verified hash is recorded on the job and advertised on
download, so the bytes can be checked end to end.
"""

import hashlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-hash-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "hash.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import train  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _upload(job_id, name, content, sha=None, extra_data=None):
    data = {"access_code": ""}
    if sha is not None:
        data["sha256"] = sha
    if extra_data:
        data.update(extra_data)
    return client.post(
        f"/api/train/{job_id}/upload",
        files={"file": (name, io.BytesIO(content), "application/octet-stream")},
        data=data,
    )


def _download(job_id, name):
    return client.get(f"/api/train/{job_id}/files/{name}", params={"access_code": ""})


def test_upload_with_matching_sha256_records_hash():
    jid = train.create_job("model-hash-ok")
    content = b"adapter-bytes-v1" * 100
    r = _upload(jid, "adapter.safetensors", content, sha=_sha(content))
    assert r.status_code == 200
    assert r.json()["sha256"] == _sha(content)
    assert r.json()["bytes"] == len(content)
    job = train.get_job(jid)
    assert job["adapter_files"] == [
        {"name": "adapter.safetensors", "sha256": _sha(content), "bytes": len(content)}
    ]
    assert _download(jid, "adapter.safetensors").content == content


def test_upload_without_sha256_still_records_server_hash():
    # Old notebooks send no hash: the file still stores, and the hash the
    # server computed while streaming is recorded either way.
    jid = train.create_job("model-hash-legacy")
    content = b"legacy-upload-no-hash"
    r = _upload(jid, "adapter.safetensors", content)
    assert r.status_code == 200
    assert r.json()["sha256"] == _sha(content)
    assert train.get_job(jid)["adapter_files"] == [
        {"name": "adapter.safetensors", "sha256": _sha(content), "bytes": len(content)}
    ]


def test_upload_with_wrong_sha256_rejected_and_old_file_kept():
    jid = train.create_job("model-hash-mismatch")
    good = b"good-adapter" * 50
    assert _upload(jid, "adapter.safetensors", good, sha=_sha(good)).status_code == 200
    # Corrupt bytes with the good file's hash claimed: 422, nothing lands.
    bad = b"corrupted-adapter" * 50
    r = _upload(jid, "adapter.safetensors", bad, sha=_sha(good))
    assert r.status_code == 422
    assert "sha256 mismatch" in r.json()["detail"]
    # The previous good adapter is untouched, bytes and record alike.
    assert _download(jid, "adapter.safetensors").content == good
    assert train.get_job(jid)["adapter_files"] == [
        {"name": "adapter.safetensors", "sha256": _sha(good), "bytes": len(good)}
    ]


def test_upload_malformed_sha256_rejected():
    jid = train.create_job("model-hash-malformed")
    r = _upload(jid, "adapter.safetensors", b"x", sha="not-a-hash")
    assert r.status_code == 400
    assert _download(jid, "adapter.safetensors").status_code == 404
    assert train.get_job(jid)["adapter_files"] == []
    r = _upload(jid, "adapter.safetensors", b"x", sha="ab" * 32 + "zz")
    assert r.status_code == 400


def test_reupload_same_name_updates_recorded_hash():
    jid = train.create_job("model-hash-reupload")
    v1 = b"adapter-v1"
    v2 = b"adapter-v2-longer"
    assert _upload(jid, "adapter.safetensors", v1, sha=_sha(v1)).status_code == 200
    assert _upload(jid, "adapter.safetensors", v2, sha=_sha(v2)).status_code == 200
    assert _download(jid, "adapter.safetensors").content == v2
    assert train.get_job(jid)["adapter_files"] == [
        {"name": "adapter.safetensors", "sha256": _sha(v2), "bytes": len(v2)}
    ]


def test_download_carries_recorded_sha256_header():
    jid = train.create_job("model-hash-header")
    content = b"verify-me" * 10
    assert _upload(jid, "a.bin", content, sha=_sha(content)).status_code == 200
    assert _upload(jid, "b.bin", b"other", sha=_sha(b"other")).status_code == 200
    r = _download(jid, "a.bin")
    assert r.status_code == 200
    assert r.headers.get("x-adapter-sha256") == _sha(content)
    r = _download(jid, "b.bin")
    assert r.headers.get("x-adapter-sha256") == _sha(b"other")


def test_list_jobs_includes_adapter_files():
    jid = train.create_job("model-hash-list")
    content = b"listed-adapter"
    assert _upload(jid, "adapter.safetensors", content, sha=_sha(content)).status_code == 200
    listed = [j for j in train.list_jobs(limit=50) if j["id"] == jid]
    assert len(listed) == 1
    assert listed[0]["adapter_files"] == [
        {"name": "adapter.safetensors", "sha256": _sha(content), "bytes": len(content)}
    ]


def test_corrupt_hashes_json_degrades_to_empty_list():
    # A hand-edited or half-written hashes column must not break reads.
    jid = train.create_job("model-hash-corrupt")
    c = train._connect()
    with train._lock:
        c.execute(
            "UPDATE train_jobs SET adapter_hashes_json=? WHERE id=?",
            ("{not json", jid),
        )
        c.commit()
    assert train.get_job(jid)["adapter_files"] == []


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("ALL ADAPTER-HASH TESTS PASS")
