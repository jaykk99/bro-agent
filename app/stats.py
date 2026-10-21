"""Usage stats — per-day chat turns per provider, in the same SQLite DB.

Lightweight observability: how much Jay uses bro-agent, on which provider,
how much context flows. No per-message content is stored here, just counts.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time

from .memory import DB_PATH

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
                "CREATE TABLE IF NOT EXISTS chat_stats ("
                "day TEXT NOT NULL, provider TEXT NOT NULL, turns INTEGER NOT NULL DEFAULT 0, "
                "in_chars INTEGER NOT NULL DEFAULT 0, out_chars INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (day, provider))"
            )
            _conn.commit()
        return _conn


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def record_turn(provider: str, in_chars: int, out_chars: int, day: str | None = None) -> None:
    day = day or _today()
    c = _connect()
    with _lock:
        c.execute(
            "INSERT INTO chat_stats (day, provider, turns, in_chars, out_chars)"
            " VALUES (?, ?, 1, ?, ?)"
            " ON CONFLICT(day, provider) DO UPDATE SET"
            " turns=turns+1, in_chars=in_chars+excluded.in_chars,"
            " out_chars=out_chars+excluded.out_chars",
            (day, provider or "unknown", max(0, in_chars), max(0, out_chars)),
        )
        c.commit()


def get_stats(days: int = 14) -> list[dict]:
    c = _connect()
    rows = c.execute(
        "SELECT day, provider, turns, in_chars, out_chars FROM chat_stats"
        " ORDER BY day DESC, provider ASC LIMIT ?",
        (max(1, days) * 4,),
    ).fetchall()
    return [dict(r) for r in rows]
