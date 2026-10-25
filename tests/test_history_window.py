"""Regression tests: chat history must return the NEWEST messages.

Before this fix, get_history() ran ORDER BY created_at ASC LIMIT 60, so
once a thread passed 60 messages the model only ever saw the thread's
opening — recent turns never reached the context window, and
build_window() could only trim within that stale slice.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory  # noqa: E402
from app.context import build_window  # noqa: E402


def _use_fresh_db(tmp_path):
    """Point memory at a throwaway DB; returns a restore callable."""
    old_path, old_conn = memory.DB_PATH, memory._conn
    if old_conn is not None:
        try:
            old_conn.close()
        except Exception:
            pass
    memory._conn = None
    memory.DB_PATH = os.path.join(str(tmp_path), "hist.db")

    def restore():
        if memory._conn is not None:
            try:
                memory._conn.close()
            except Exception:
                pass
        memory._conn = None
        memory.DB_PATH = old_path

    return restore


def _fill(tid, n):
    for i in range(n):
        memory.add_message(tid, "user", f"msg{i}")


def test_history_returns_newest_not_oldest(tmp_path):
    restore = _use_fresh_db(tmp_path)
    try:
        tid = memory.create_thread("long")
        _fill(tid, 70)
        hist = memory.get_history(tid)  # default limit 60
        assert len(hist) == 60
        assert hist[0]["content"] == "msg10"
        assert hist[-1]["content"] == "msg69"
        contents = [m["content"] for m in hist]
        assert contents == sorted(contents, key=lambda c: int(c[3:]))
    finally:
        restore()


def test_history_limit_returns_newest_slice(tmp_path):
    restore = _use_fresh_db(tmp_path)
    try:
        tid = memory.create_thread("sliced")
        _fill(tid, 10)
        hist = memory.get_history(tid, limit=5)
        assert [m["content"] for m in hist] == ["msg5", "msg6", "msg7", "msg8", "msg9"]
    finally:
        restore()


def test_history_short_thread_unchanged(tmp_path):
    restore = _use_fresh_db(tmp_path)
    try:
        tid = memory.create_thread("short")
        _fill(tid, 3)
        assert [m["content"] for m in memory.get_history(tid)] == ["msg0", "msg1", "msg2"]
    finally:
        restore()


def test_history_identical_timestamps_keep_insertion_order(tmp_path, monkeypatch):
    restore = _use_fresh_db(tmp_path)
    monkeypatch.setattr(memory.time, "time", lambda: 1234.0)
    try:
        tid = memory.create_thread("ties")
        _fill(tid, 5)
        assert [m["content"] for m in memory.get_history(tid)] == [
            "msg0", "msg1", "msg2", "msg3", "msg4",
        ]
    finally:
        restore()


def test_window_over_fixed_history_sees_recent_turns(tmp_path):
    restore = _use_fresh_db(tmp_path)
    try:
        tid = memory.create_thread("windowed")
        _fill(tid, 70)
        window = build_window(memory.get_history(tid), max_chars=10_000)
        contents = [m["content"] for m in window if m["role"] != "system"]
        assert contents[-1] == "msg69"
        assert "msg0" not in contents
    finally:
        restore()
