"""Train adapter uploads must be atomic: a rejected re-upload must not
destroy the good file it was replacing, and a download must never see a
partial upload.

Proven gap (2026-10-31, before the fix): the upload route opened the
final destination with "wb" and streamed into it, so
  * a re-upload that tripped the per-file/per-job cap truncated the old
    file first, then the 413 handler unlinked it — the previously good
    adapter was gone, not just the rejected bytes;
  * a download running concurrently with an upload served the truncated,
    half-written file as if it were a complete adapter.
The fix streams into a temp file in the same job dir and os.replace()s it
into place only after the whole upload fits under both caps.
"""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-atomic-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "atomic.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

from fastapi.testclient import TestClient  # noqa: E402

from app import train  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def _upload(job_id, name, content):
    return client.post(
        f"/api/train/{job_id}/upload",
        files={"file": (name, io.BytesIO(content), "application/octet-stream")},
        data={"access_code": ""},
    )


def _download(job_id, name):
    return client.get(f"/api/train/{job_id}/files/{name}", params={"access_code": ""})


def test_over_cap_reupload_keeps_old_file(monkeypatch):
    # Job cap 2500: a.bin (1000) + b.bin (1000) stored. Re-uploading
    # a.bin with 2000 bytes would push the job to 3000 > 2500, so it
    # must be rejected — and the old a.bin must survive intact.
    monkeypatch.setenv("BRO_ADAPTER_JOB_MAX_BYTES", "2500")
    jid = train.create_job("model-atomic")
    assert _upload(jid, "a.bin", b"x" * 1000).status_code == 200
    assert _upload(jid, "b.bin", b"y" * 1000).status_code == 200
    r = _upload(jid, "a.bin", b"z" * 2000)
    assert r.status_code == 413
    dest = os.path.join(train.adapter_job_dir(jid), "a.bin")
    assert os.path.isfile(dest)
    with open(dest, "rb") as f:
        assert f.read() == b"x" * 1000
    # Download serves the old file, not a truncation or nothing.
    d = _download(jid, "a.bin")
    assert d.status_code == 200
    assert d.content == b"x" * 1000
    # The rejected upload left no temp litter behind.
    leftovers = [n for n in os.listdir(train.adapter_job_dir(jid))
                 if n.startswith(".upload-")]
    assert leftovers == []
    assert train.adapter_total_bytes(jid) == 2000


def test_failed_new_upload_leaves_no_temp(monkeypatch):
    monkeypatch.setenv("BRO_ADAPTER_JOB_MAX_BYTES", "1500")
    jid = train.create_job("model-atomic-new")
    assert _upload(jid, "a.bin", b"x" * 1000).status_code == 200
    r = _upload(jid, "b.bin", b"y" * 1000)
    assert r.status_code == 413
    d = train.adapter_job_dir(jid)
    assert not os.path.exists(os.path.join(d, "b.bin"))
    assert [n for n in os.listdir(d) if n.startswith(".upload-")] == []
    assert train.adapter_total_bytes(jid) == 1000


def test_temp_upload_names_are_not_downloadable(monkeypatch):
    # The temp name pattern must never resolve to a servable file: a
    # download for a planted temp-looking name 404s instead of serving
    # whatever sits there mid-upload.
    jid = train.create_job("model-atomic-temp")
    dest_dir = train.adapter_job_dir(jid)
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, ".upload-deadbeef.tmp"), "wb") as f:
        f.write(b"PARTIAL")
    r = _download(jid, ".upload-deadbeef.tmp")
    assert r.status_code == 404
    assert b"PARTIAL" not in r.content
