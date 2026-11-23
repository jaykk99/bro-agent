"""Pin the remaining untested paths in app/train.py.

Coverage gaps left after test_train.py / test_train_upload_*.py:
corrupt metrics JSON in _row_to_dict, complete_job's bad-status
ValueError, record_adapter_file's unknown-job / corrupt / non-dict
hash JSON, the adapters_total_bytes / adapter_total_bytes OSError
continues, and open_adapter_job_dir / open_adapter_file's fstat,
readlink and escape-check branches. Real dirs and files where the
trigger is real; fault injection only for the TOCTOU races.
"""

import json
import os
import stat
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-gaps-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

from app import train  # noqa: E402


def _set_col(jid: str, col: str, value: str) -> None:
    c = train._connect()
    with train._lock:
        c.execute(f"UPDATE train_jobs SET {col}=? WHERE id=?", (value, jid))
        c.commit()


# ---- _row_to_dict: corrupt metrics JSON ----


def test_row_to_dict_corrupt_metrics_degrades_to_none():
    jid = train.create_job("model-metrics")
    # Write a metrics blob that is not JSON straight into the row.
    _set_col(jid, "metrics_json", "{not-json")
    job = train.get_job(jid)
    assert job["metrics"] is None


# ---- complete_job: bad status ----


def test_complete_job_rejects_bad_status():
    with pytest.raises(ValueError, match="status must be done or failed"):
        train.complete_job("whatever", "weird")


# ---- record_adapter_file: unknown job / corrupt / non-dict ----


def test_record_adapter_file_unknown_job_is_noop():
    train.record_adapter_file("no-such-job", "a.safetensors", "ab" * 32, 10)
    assert train.get_job("no-such-job") is None


def test_record_adapter_file_recovers_from_corrupt_hash_json():
    jid = train.create_job("model-corrupt")
    _set_col(jid, "adapter_hashes_json", "{broken")
    train.record_adapter_file(jid, "adapter_model.safetensors", "cd" * 32, 123)
    files = train.get_job(jid)["adapter_files"]
    assert files == [{"name": "adapter_model.safetensors", "sha256": "cd" * 32, "bytes": 123}]


def test_record_adapter_file_recovers_from_non_dict_hash_json():
    jid = train.create_job("model-nondict")
    _set_col(jid, "adapter_hashes_json", "[1, 2]")
    train.record_adapter_file(jid, "adapter_model.safetensors", "ef" * 32, 456)
    files = train.get_job(jid)["adapter_files"]
    assert files == [{"name": "adapter_model.safetensors", "sha256": "ef" * 32, "bytes": 456}]


# ---- adapters_total_bytes: missing dir / entry errors ----


def test_adapters_total_bytes_missing_dir_is_zero(monkeypatch):
    monkeypatch.setattr(train, "ADAPTERS_DIR", "/nonexistent-adapters-dir-xyz")
    assert train.adapters_total_bytes() == 0


class _RaisingEntry:
    """A DirEntry whose stat calls blow up (file vanished mid-scan)."""

    name = "adapter_model.safetensors"
    path = "/fake/adapter_model.safetensors"
    called = False

    def is_dir(self, follow_symlinks=False):
        return True

    def is_file(self, follow_symlinks=False):
        type(self).called = True
        raise OSError("stat down")

    def stat(self, follow_symlinks=False):
        raise OSError("stat down")


class _CtxList:
    def __init__(self, items):
        self._items = items

    def __enter__(self):
        return self._items

    def __exit__(self, *a):
        return False


def test_adapters_total_bytes_entry_stat_error_is_skipped(monkeypatch):
    jid = train.create_job("model-scan")
    job_dir = os.path.dirname(train.adapter_file_path(jid, "x", create_dir=True))
    real_scandir = os.scandir

    def fake_scandir(path, *a, **k):
        if os.fspath(path) == job_dir:
            return _CtxList([_RaisingEntry()])
        return real_scandir(path, *a, **k)

    _RaisingEntry.called = False
    monkeypatch.setattr(os, "scandir", fake_scandir)
    total = train.adapters_total_bytes()
    assert isinstance(total, int) and total >= 0
    assert _RaisingEntry.called  # the raising entry was really visited


def test_adapters_total_bytes_job_dir_scan_error_is_skipped(monkeypatch):
    jid = train.create_job("model-scan2")
    job_dir = os.path.dirname(train.adapter_file_path(jid, "x", create_dir=True))
    real_scandir = os.scandir
    seen = {"n": 0}

    def fake_scandir(path, *a, **k):
        if os.fspath(path) == job_dir:
            seen["n"] += 1
            raise OSError("dir vanished")
        return real_scandir(path, *a, **k)

    monkeypatch.setattr(os, "scandir", fake_scandir)
    total = train.adapters_total_bytes()
    assert isinstance(total, int) and total >= 0
    assert seen["n"] == 1


def test_adapter_total_bytes_entry_stat_error_is_skipped(monkeypatch):
    jid = train.create_job("model-scan3")
    d = os.path.dirname(train.adapter_file_path(jid, "x", create_dir=True))
    real_scandir = os.scandir

    def fake_scandir(path, *a, **k):
        if os.fspath(path) == d:
            return _CtxList([_RaisingEntry()])
        return real_scandir(path, *a, **k)

    _RaisingEntry.called = False
    monkeypatch.setattr(os, "scandir", fake_scandir)
    assert train.adapter_total_bytes(jid) == 0
    assert _RaisingEntry.called


# ---- open_adapter_job_dir: fstat / readlink / escape ----


def test_open_adapter_job_dir_non_directory(monkeypatch):
    jid = train.create_job("model-opendir")
    train.adapter_file_path(jid, "x", create_dir=True)
    # fstat says "not a directory": the BaseException handler must close
    # the fd and re-raise instead of leaking it.
    monkeypatch.setattr(stat, "S_ISDIR", lambda mode: False)
    with pytest.raises(FileNotFoundError, match="not a directory"):
        train.open_adapter_job_dir(jid)


def test_open_adapter_job_dir_readlink_failure_degrades(monkeypatch):
    jid = train.create_job("model-readlink")
    train.adapter_file_path(jid, "x", create_dir=True)
    real_readlink = os.readlink

    def flaky_readlink(path, *a, **k):
        if str(path).startswith("/proc/self/fd/"):
            raise OSError("no proc")
        return real_readlink(path, *a, **k)

    monkeypatch.setattr(os, "readlink", flaky_readlink)
    fd = train.open_adapter_job_dir(jid)
    try:
        assert os.fstat(fd).st_ino > 0
    finally:
        os.close(fd)


def test_open_adapter_job_dir_escape_is_refused(monkeypatch):
    jid = train.create_job("model-escape")
    train.adapter_file_path(jid, "x", create_dir=True)
    real_readlink = os.readlink

    def lying_readlink(path, *a, **k):
        if str(path).startswith("/proc/self/fd/"):
            return "/etc"
        return real_readlink(path, *a, **k)

    monkeypatch.setattr(os, "readlink", lying_readlink)
    with pytest.raises(FileNotFoundError, match="escapes the adapters directory"):
        train.open_adapter_job_dir(jid)


# ---- open_adapter_file: missing file / readlink / escape ----


def test_open_adapter_file_missing_file():
    jid = train.create_job("model-openfile")
    train.adapter_file_path(jid, "x", create_dir=True)
    with pytest.raises(FileNotFoundError, match="adapter file not found"):
        train.open_adapter_file(jid, "nope.safetensors")


def test_open_adapter_file_readlink_failure_degrades(monkeypatch):
    jid = train.create_job("model-openfile2")
    dest = train.adapter_file_path(jid, "adapter_model.safetensors", create_dir=True)
    with open(dest, "wb") as f:
        f.write(b"adapter-bytes")
    real_readlink = os.readlink

    def flaky_readlink(path, *a, **k):
        if str(path).startswith("/proc/self/fd/"):
            raise OSError("no proc")
        return real_readlink(path, *a, **k)

    monkeypatch.setattr(os, "readlink", flaky_readlink)
    fd, size, name = train.open_adapter_file(jid, "adapter_model.safetensors")
    try:
        assert size == len(b"adapter-bytes")
        assert name == "adapter_model.safetensors"
    finally:
        os.close(fd)


def test_open_adapter_file_escape_is_refused(monkeypatch):
    jid = train.create_job("model-openfile3")
    dest = train.adapter_file_path(jid, "adapter_model.safetensors", create_dir=True)
    with open(dest, "wb") as f:
        f.write(b"adapter-bytes")
    real_readlink = os.readlink

    def lying_readlink(path, *a, **k):
        if str(path).startswith("/proc/self/fd/"):
            return "/etc/passwd"
        return real_readlink(path, *a, **k)

    monkeypatch.setattr(os, "readlink", lying_readlink)
    with pytest.raises(FileNotFoundError, match="escapes the adapters directory"):
        train.open_adapter_file(jid, "adapter_model.safetensors")
