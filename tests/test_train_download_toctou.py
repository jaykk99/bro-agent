"""Train adapter download TOCTOU fix (2026-11-03).

The download route used to check islink/isfile on the path and then hand
that path to FileResponse, which opened it later: a symlink swapped in
between the check and the open made the route serve the link target
outside the adapters tree. The route now streams from a descriptor
opened with O_NOFOLLOW (job dir via dir fd, file relative to it), so
the served bytes are pinned to the inode the open resolved.

These tests prove the gap is closed, not just the old symlink cases:
  * a planted symlink (file or job dir) still 404s;
  * a path swapped after the open still serves the ORIGINAL bytes.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-download-toctou-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "download-toctou.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import train  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def _write_adapter(jid, name, content):
    d = train.adapter_job_dir(jid)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(content)
    return p


def test_download_serves_exact_bytes_and_headers():
    jid = train.create_job("model-dl-ok")
    _write_adapter(jid, "adapter.bin", b"ADAPTER-BYTES-123")
    r = client.get(f"/api/train/{jid}/files/adapter.bin", params={"access_code": ""})
    assert r.status_code == 200
    assert r.content == b"ADAPTER-BYTES-123"
    assert r.headers["content-length"] == "17"
    assert "adapter.bin" in r.headers["content-disposition"]


def test_download_symlinked_file_404s(tmp_path):
    jid = train.create_job("model-dl-link")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SECRET")
    d = train.adapter_job_dir(jid)
    os.makedirs(d, exist_ok=True)
    os.symlink(str(outside), os.path.join(d, "evil.bin"))
    r = client.get(f"/api/train/{jid}/files/evil.bin", params={"access_code": ""})
    assert r.status_code == 404
    assert b"OUTSIDE-SECRET" not in r.content


def test_download_symlinked_job_dir_404s(tmp_path):
    jid = train.create_job("model-dl-dirlink")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "adapter.bin").write_text("OUTSIDE-SECRET")
    link_dir = train.adapter_job_dir(jid)
    os.symlink(str(outside_dir), link_dir)
    r = client.get(f"/api/train/{jid}/files/adapter.bin", params={"access_code": ""})
    assert r.status_code == 404
    assert b"OUTSIDE-SECRET" not in r.content


def test_download_temp_name_404s():
    jid = train.create_job("model-dl-temp")
    _write_adapter(jid, ".upload-abc123.tmp", b"TEMP-BYTES")
    r = client.get(
        f"/api/train/{jid}/files/.upload-abc123.tmp", params={"access_code": ""}
    )
    assert r.status_code == 404
    assert b"TEMP-BYTES" not in r.content


def test_open_pins_inode_across_path_swap():
    """The core TOCTOU proof: open, then replace the path (the way an
    upload's os.replace — or a planted symlink swap — would), and the
    open descriptor must still read the ORIGINAL bytes, never the new
    path's content. A path-based serve would read the replacement."""
    jid = train.create_job("model-dl-pin")
    p = _write_adapter(jid, "adapter.bin", b"ORIGINAL-ADAPTER")
    fd, size, name = train.open_adapter_file(jid, "adapter.bin")
    assert size == len(b"ORIGINAL-ADAPTER")
    assert name == "adapter.bin"
    try:
        # Swap the path out from under the open descriptor.
        with open(p + ".new", "wb") as f:
            f.write(b"REPLACEMENT-ADAPTER")
        os.replace(p + ".new", p)
        assert open(p, "rb").read() == b"REPLACEMENT-ADAPTER"
        assert os.read(fd, 1024) == b"ORIGINAL-ADAPTER"
    finally:
        os.close(fd)


def test_open_refuses_symlink_and_nonregular(tmp_path):
    jid = train.create_job("model-dl-open")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SECRET")
    d = train.adapter_job_dir(jid)
    os.makedirs(d, exist_ok=True)
    os.symlink(str(outside), os.path.join(d, "evil.bin"))
    with pytest.raises(FileNotFoundError):
        train.open_adapter_file(jid, "evil.bin")
    os.makedirs(os.path.join(d, "subdir"), exist_ok=True)
    with pytest.raises(FileNotFoundError):
        train.open_adapter_file(jid, "subdir")
    with pytest.raises(FileNotFoundError):
        train.open_adapter_file("bogus-job-dl-open-xyz", "nope.bin")
