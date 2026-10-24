"""Persistent memory — SQLite-backed threads and messages.

Single-user personal agent. Threads hold conversations; every turn is
stored so context persists across sessions and restarts.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid

DB_PATH = os.environ.get("BRO_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "bro.db"))

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute(
                "CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, title TEXT, created_at REAL)"
            )
            _conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, role TEXT NOT NULL, "
                "content TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            _conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at)")
            _conn.commit()
        return _conn


def create_thread(title: str = "New chat") -> str:
    tid = uuid.uuid4().hex[:12]
    c = _connect()
    with _lock:
        c.execute("INSERT INTO threads (id, title, created_at) VALUES (?, ?, ?)", (tid, title, time.time()))
        c.commit()
    return tid


def list_threads(limit: int = 50) -> list[dict]:
    c = _connect()
    rows = c.execute("SELECT id, title, created_at FROM threads ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def add_message(thread_id: str, role: str, content: str) -> None:
    c = _connect()
    with _lock:
        c.execute(
            "INSERT INTO messages (id, thread_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, thread_id, role, content, time.time()),
        )
        c.commit()


def get_history(thread_id: str, limit: int = 60) -> list[dict]:
    """Chronological message history for a thread (oldest first)."""
    c = _connect()
    rows = c.execute(
        "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY created_at ASC LIMIT ?",
        (thread_id, limit),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_thread_messages(thread_id: str, limit: int = 200, offset: int = 0) -> list[dict]:
    """Full messages for the UI, with timestamps (oldest first).

    `offset` skips the oldest messages first — the UI loads page 1 of the
    oldest messages, then advances the offset.
    """
    c = _connect()
    rows = c.execute(
        "SELECT role, content, created_at FROM messages WHERE thread_id = ?"
        " ORDER BY created_at ASC LIMIT ? OFFSET ?",
        (thread_id, max(1, limit), max(0, offset)),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]


def rename_thread(thread_id: str, title: str) -> bool:
    title = (title or "").strip()[:60]
    if not title:
        return False
    c = _connect()
    with _lock:
        cur = c.execute("UPDATE threads SET title=? WHERE id=?", (title, thread_id))
        c.commit()
        return cur.rowcount > 0


def delete_thread(thread_id: str) -> bool:
    c = _connect()
    with _lock:
        c.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        cur = c.execute("DELETE FROM threads WHERE id=?", (thread_id,))
        c.commit()
        return cur.rowcount > 0


def thread_exists(thread_id: str) -> bool:
    c = _connect()
    row = c.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone()
    return row is not None


def search_threads(query: str, limit: int = 50) -> list[dict]:
    """Search threads by title or message content, most recent first.

    Empty query falls back to the plain list (same as /api/threads).
    """
    q = (query or "").strip()
    if not q:
        return list_threads(limit)
    like = f"%{q}%"
    c = _connect()
    rows = c.execute(
        "SELECT DISTINCT t.id, t.title, t.created_at FROM threads t "
        "LEFT JOIN messages m ON m.thread_id = t.id "
        "WHERE t.title LIKE ? OR m.content LIKE ? "
        "ORDER BY t.created_at DESC LIMIT ?",
        (like, like, max(1, limit)),
    ).fetchall()
    return [dict(r) for r in rows]
