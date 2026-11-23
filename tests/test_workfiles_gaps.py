"""Pin the remaining untested paths in app/workfiles.py.

Coverage gaps left after test_workfiles.py: the list_dir lstat
OSError continue, the MAX_LIST_ENTRIES cap, the open_work_file
fstat-OSError and is-a-directory branches, and the delete_work_file
not-a-regular-file branch. Real dirs and files; fault injection
only for the stat races.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import workfiles  # noqa: E402


@pytest.fixture()
def root(tmp_path):
    r = tmp_path / "work"
    r.mkdir()
    return str(r)


def test_list_dir_skips_unstatable_entry(root, monkeypatch):
    with open(os.path.join(root, "good.txt"), "w") as f:
        f.write("ok")
    with open(os.path.join(root, "bad.txt"), "w") as f:
        f.write("gone")
    real_lstat = os.lstat

    def flaky_lstat(path, *a, **k):
        if os.path.basename(os.fspath(path)) == "bad.txt":
            raise OSError("vanished")
        return real_lstat(path, *a, **k)

    monkeypatch.setattr(os, "lstat", flaky_lstat)
    names = [e["name"] for e in workfiles.list_dir(root)]
    assert names == ["good.txt"]


def test_list_dir_caps_entries(root, monkeypatch):
    for i in range(5):
        with open(os.path.join(root, f"f{i}.txt"), "w") as f:
            f.write("x")
    monkeypatch.setattr(workfiles, "MAX_LIST_ENTRIES", 2)
    assert len(workfiles.list_dir(root)) == 2


def test_open_work_file_fstat_failure_closes_fd_and_raises(root, monkeypatch):
    with open(os.path.join(root, "a.txt"), "w") as f:
        f.write("data")

    def boom(fd):
        raise OSError("fstat down")

    monkeypatch.setattr(os, "fstat", boom)
    with pytest.raises(OSError, match="fstat down"):
        workfiles.open_work_file(root, "a.txt")


def test_open_work_file_directory_is_rejected(root):
    os.makedirs(os.path.join(root, "subdir"))
    with pytest.raises(IsADirectoryError):
        workfiles.open_work_file(root, "subdir")


def test_open_work_file_fifo_is_rejected(root):
    # resolve() follows a pre-existing in-tree symlink by design, so
    # the not-a-regular-file branch needs a FIFO: a writer thread lets
    # the open succeed, then fstat reports S_ISFIFO, not S_ISREG.
    import threading

    path = os.path.join(root, "pipe")
    os.mkfifo(path)

    def writer():
        try:
            fd = os.open(path, os.O_WRONLY)
            os.close(fd)
        except OSError:
            pass

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            workfiles.open_work_file(root, "pipe")
    finally:
        t.join(timeout=5)


def test_delete_work_file_fifo_is_rejected(root):
    # A FIFO is not a regular file: the stat-only delete path refuses
    # it with ValueError instead of unlinking something it cannot serve.
    os.mkfifo(os.path.join(root, "pipe"))
    with pytest.raises(ValueError, match="not a regular file"):
        workfiles.delete_work_file(root, "pipe")
    assert os.path.exists(os.path.join(root, "pipe"))
