"""Train adapter upload/download hardening — per-job cap + symlink refusal.

Proven gaps (2026-10-30, before the fix):
  * a symlink planted at adapters/<job>/<file> made the download route
    serve, and the upload route overwrite, the link target outside the
    adapters tree — while the generic /api/upload route already refused
    symlinks via sandbox.resolve().
  * the 2 GB per-file cap never bounded a job: unlimited files per job
    were accepted, so one job could still fill the disk.
"""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-caps-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "caps.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import train  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("BRO_WORKDIR", str(work))
    return work


def _upload(job_id, name, content):
    return client.post(
        f"/api/train/{job_id}/upload",
        files={"file": (name, io.BytesIO(content), "application/octet-stream")},
        data={"access_code": ""},
    )


def test_download_symlink_is_not_served(tmp_path):
    jid = train.create_job("model-symlink")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SECRET")
    dest_dir = train.adapter_job_dir(jid)
    os.makedirs(dest_dir, exist_ok=True)
    link = os.path.join(dest_dir, "evil.bin")
    os.symlink(str(outside), link)
    r = client.get(f"/api/train/{jid}/files/evil.bin", params={"access_code": ""})
    assert r.status_code == 404
    assert b"OUTSIDE-SECRET" not in r.content


def test_upload_over_symlink_refused_and_target_untouched(tmp_path):
    jid = train.create_job("model-symlink-up")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SECRET")
    dest_dir = train.adapter_job_dir(jid)
    os.makedirs(dest_dir, exist_ok=True)
    os.symlink(str(outside), os.path.join(dest_dir, "evil.bin"))
    r = _upload(jid, "evil.bin", b"PWNED")
    assert r.status_code == 400
    assert outside.read_text() == "OUTSIDE-SECRET"
    assert os.path.islink(os.path.join(dest_dir, "evil.bin"))


def test_download_bogus_job_creates_no_directory():
    r = client.get("/api/train/bogus-job-caps-xyz/files/nope.bin", params={"access_code": ""})
    assert r.status_code == 404
    assert not os.path.exists(os.path.join(train.ADAPTERS_DIR, "bogus-job-caps-xyz"))


def test_per_job_total_cap(monkeypatch):
    # Tighten only the per-job cap for this test (per-file cap stays
    # huge, so only the job total can trip).
    monkeypatch.setenv("BRO_ADAPTER_JOB_MAX_BYTES", "2500")
    jid = train.create_job("model-cap")
    assert _upload(jid, "a.bin", b"x" * 1000).status_code == 200
    assert _upload(jid, "b.bin", b"y" * 1000).status_code == 200
    r = _upload(jid, "c.bin", b"z" * 1000)
    assert r.status_code == 413
    # Rejected upload leaves nothing behind; the first two survive.
    assert not os.path.exists(os.path.join(train.adapter_job_dir(jid), "c.bin"))
    assert train.adapter_total_bytes(jid) == 2000


def test_reupload_same_name_does_not_double_count(monkeypatch):
    monkeypatch.setenv("BRO_ADAPTER_JOB_MAX_BYTES", "2500")
    jid = train.create_job("model-recap")
    assert _upload(jid, "a.bin", b"x" * 1000).status_code == 200
    # Replacing a.bin with a same-size file must fit under the cap even
    # though 1000 bytes are already stored for this job.
    assert _upload(jid, "a.bin", b"x" * 1000).status_code == 200
    assert train.adapter_total_bytes(jid) == 1000
