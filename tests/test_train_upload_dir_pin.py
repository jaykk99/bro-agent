"""Train adapter uploads must stay inside the adapters tree even when
the job dir is swapped for a symlink mid-upload.

Proven gap (2026-11-04, before the fix): the upload route created its
temp file by path (open(tmp) in the job dir) and landed it by path
(os.replace(tmp, dest)), and the symlink re-check just before the
replace only inspected the path at that instant. A job dir swapped for
a symlink after adapter_file_path()'s realpath check — or after that
re-check — made BOTH operations resolve through the link, so the temp
file was created, and the adapter landed, in the link target outside
the adapters tree. The old "replace swaps the link, never its target"
reasoning held for a symlinked FILE, not for a symlinked DIR, where
both paths resolve through the link.

The fix opens the job dir once with O_DIRECTORY | O_NOFOLLOW
(train.open_adapter_job_dir) and creates, writes, unlinks, and renames
strictly relative to that fd — a path swap after the open cannot
redirect any of them, because none of them re-resolves the path. The
temp create additionally uses O_CREAT | O_EXCL | O_NOFOLLOW, so a temp
name planted as a symlink fails instead of being followed.

These tests prove the gap is closed, not just the old symlink cases:
  * an already-symlinked job dir is refused (400) and the link target
    stays untouched;
  * a dir fd opened BEFORE a path swap still renames inside the
    original dir — the exact mechanism the route now relies on — while
    a path-based open after the same swap follows the link;
  * a temp name planted as a symlink fails under the route's
    O_CREAT | O_EXCL | O_NOFOLLOW flags instead of being followed;
  * normal uploads (including a re-upload replacing an old adapter)
    still land byte-exact through the pinned path.
"""

import io
import os
import stat
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-dirpin-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "dirpin.db")
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


def test_symlinked_job_dir_upload_refused_and_target_untouched(tmp_path):
    jid = train.create_job("model-dirpin-link")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "planted.bin"
    target.write_bytes(b"OUTSIDE-SECRET")
    d = train.adapter_job_dir(jid)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    os.symlink(str(outside), d)
    r = _upload(jid, "adapter.bin", b"UPLOADED-BYTES")
    assert r.status_code == 400
    # Nothing landed through the link: no new file in the target dir,
    # and the planted file is byte-identical.
    assert sorted(os.listdir(outside)) == ["planted.bin"]
    assert target.read_bytes() == b"OUTSIDE-SECRET"


def test_symlinked_dest_file_upload_refused(tmp_path):
    jid = train.create_job("model-dirpin-filelink")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SECRET")
    d = train.adapter_job_dir(jid)
    os.makedirs(d, exist_ok=True)
    os.symlink(str(outside), os.path.join(d, "adapter.bin"))
    r = _upload(jid, "adapter.bin", b"UPLOADED-BYTES")
    assert r.status_code == 400
    assert outside.read_text() == "OUTSIDE-SECRET"


def test_open_adapter_job_dir_refuses_symlink(tmp_path):
    jid = train.create_job("model-dirpin-opendir")
    outside = tmp_path / "outside"
    outside.mkdir()
    d = train.adapter_job_dir(jid)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    os.symlink(str(outside), d)
    try:
        train.open_adapter_job_dir(jid)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("symlinked job dir must fail to open")
    # And a real dir opens to a directory fd.
    jid2 = train.create_job("model-dirpin-opendir2")
    d2 = train.adapter_job_dir(jid2)
    os.makedirs(d2, exist_ok=True)
    fd = train.open_adapter_job_dir(jid2)
    try:
        assert stat.S_ISDIR(os.fstat(fd).st_mode)
    finally:
        os.close(fd)


def test_dir_fd_rename_survives_path_swap(tmp_path):
    """The mechanism the route relies on: a dir fd opened before a
    path swap still renames inside the ORIGINAL dir, while a
    path-based open after the same swap follows the link. If this ever
    stops holding, the route's pinning stops meaning anything."""
    jid = train.create_job("model-dirpin-swap")
    d = train.adapter_job_dir(jid)
    os.makedirs(d, exist_ok=True)
    dir_fd = train.open_adapter_job_dir(jid)
    try:
        tmp_fd = os.open(
            ".upload-probe.tmp",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=dir_fd,
        )
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(b"PINNED-BYTES")
        # Swap the path for a symlink to an outside dir (rename the
        # real dir aside, then plant the link at its path).
        outside = tmp_path / "outside"
        outside.mkdir()
        os.rename(d, d + ".orig")
        os.symlink(str(outside), d)
        # Path-based access now follows the link (the old bug's shape).
        assert os.path.realpath(os.path.join(d, ".upload-probe.tmp")) != os.path.join(
            d, ".upload-probe.tmp"
        ) or not os.path.exists(os.path.join(d, ".upload-probe.tmp"))
        # The fd-relative rename lands in the original dir, not the target.
        os.rename(".upload-probe.tmp", "adapter.bin", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        assert sorted(os.listdir(outside)) == []
        # d now resolves through the link, so read via the fd instead.
        fd = os.open("adapter.bin", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd)
        try:
            assert os.read(fd, 100) == b"PINNED-BYTES"
        finally:
            os.close(fd)
    finally:
        os.close(dir_fd)


def test_planted_symlink_temp_name_fails_under_route_flags():
    """A temp name planted as a symlink must fail under
    O_CREAT | O_EXCL | O_NOFOLLOW instead of being followed to the
    link target — the flags the route creates its temp file with."""
    jid = train.create_job("model-dirpin-tmplink")
    d = train.adapter_job_dir(jid)
    os.makedirs(d, exist_ok=True)
    outside = os.path.join(d, "..", "tmplink-target.txt")
    with open(os.path.join(d, "tmplink-target.txt"), "wb") as f:
        f.write(b"TARGET")
    dir_fd = train.open_adapter_job_dir(jid)
    try:
        os.symlink("tmplink-target.txt", os.path.join(d, ".upload-planted.tmp"))
        try:
            os.open(
                ".upload-planted.tmp",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=dir_fd,
            )
        except OSError:
            pass
        else:
            raise AssertionError("planted symlink temp name must fail to open")
        with open(os.path.join(d, "tmplink-target.txt"), "rb") as f:
            assert f.read() == b"TARGET"
    finally:
        os.close(dir_fd)


def test_pinned_upload_and_reupload_still_land():
    jid = train.create_job("model-dirpin-ok")
    assert _upload(jid, "adapter.bin", b"FIRST-BYTES").status_code == 200
    d = _download(jid, "adapter.bin")
    assert d.status_code == 200
    assert d.content == b"FIRST-BYTES"
    assert _upload(jid, "adapter.bin", b"SECOND-BYTES!!").status_code == 200
    d = _download(jid, "adapter.bin")
    assert d.content == b"SECOND-BYTES!!"
    leftovers = [n for n in os.listdir(train.adapter_job_dir(jid))
                 if n.startswith(".upload-")]
    assert leftovers == []
