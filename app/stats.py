"""Usage stats — per-day chat turns per provider, in the same SQLite DB.

Lightweight observability: how much Jay uses bro-agent, on which provider,
how much context flows, how often a provider fails or refuses a turn, and
how long successful turns take. No per-message content is stored here, just
counts and timings.
"""

from __future__ import annotations

import datetime
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
                "failures INTEGER NOT NULL DEFAULT 0, total_ms INTEGER NOT NULL DEFAULT 0, "
                "fail_ms INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (day, provider))"
            )
            # Migrate DBs created before failures/total_ms existed: the
            # CREATE TABLE above is a no-op for them, so add the columns.
            _cols = {r[1] for r in _conn.execute("PRAGMA table_info(chat_stats)")}
            if "failures" not in _cols:
                _conn.execute("ALTER TABLE chat_stats ADD COLUMN failures INTEGER NOT NULL DEFAULT 0")
            if "total_ms" not in _cols:
                _conn.execute("ALTER TABLE chat_stats ADD COLUMN total_ms INTEGER NOT NULL DEFAULT 0")
            if "fail_ms" not in _cols:
                # Failed-attempt latency: how long a provider burned before
                # it failed (queue timeout, error, or refusal on legit
                # work). Without it a failure is a bare count and a
                # 60s Horde queue timeout looks identical to an instant
                # 403 — ops cannot tell a slow outage from a fast one.
                _conn.execute("ALTER TABLE chat_stats ADD COLUMN fail_ms INTEGER NOT NULL DEFAULT 0")
            _conn.commit()
        return _conn


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def record_turn(
    provider: str,
    in_chars: int,
    out_chars: int,
    day: str | None = None,
    duration_ms: int = 0,
) -> None:
    day = day or _today()
    c = _connect()
    with _lock:
        c.execute(
            "INSERT INTO chat_stats (day, provider, turns, in_chars, out_chars, total_ms)"
            " VALUES (?, ?, 1, ?, ?, ?)"
            " ON CONFLICT(day, provider) DO UPDATE SET"
            " turns=turns+1, in_chars=in_chars+excluded.in_chars,"
            " out_chars=out_chars+excluded.out_chars,"
            " total_ms=total_ms+excluded.total_ms",
            (day, provider or "unknown", max(0, in_chars), max(0, out_chars), max(0, duration_ms)),
        )
        c.commit()


def record_failure(provider: str, day: str | None = None, duration_ms: int = 0) -> None:
    """Count one failed turn (hard error, or a refusal on legit work that
    made the router fall through to the next provider). A day with only
    failures still shows up in the ops dashboard, so an outage is visible
    instead of looking like no usage. duration_ms is how long THIS
    attempt burned before it failed — a failed provider's latency is
    recorded against the failure (fail_ms), never folded into the next
    provider's successful-turn average (see record_turn)."""
    day = day or _today()
    c = _connect()
    with _lock:
        c.execute(
            "INSERT INTO chat_stats (day, provider, turns, in_chars, out_chars, failures, total_ms, fail_ms)"
            " VALUES (?, ?, 0, 0, 0, 1, 0, ?)"
            " ON CONFLICT(day, provider) DO UPDATE SET"
            " failures=failures+1, fail_ms=fail_ms+excluded.fail_ms",
            (day, provider or "unknown", max(0, duration_ms)),
        )
        c.commit()


def get_stats(days: int = 14) -> list[dict]:
    c = _connect()
    days = max(1, days)
    cutoff = (datetime.date.today() - datetime.timedelta(days=days - 1)).isoformat()
    rows = c.execute(
        "SELECT day, provider, turns, in_chars, out_chars, failures, total_ms, fail_ms"
        " FROM chat_stats WHERE day >= ? ORDER BY day DESC, provider ASC",
        (cutoff,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["avg_ms"] = round(d["total_ms"] / d["turns"]) if d["turns"] else 0
        d["avg_fail_ms"] = round(d["fail_ms"] / d["failures"]) if d["failures"] else 0
        out.append(d)
    return out


def get_summary(days: int = 14) -> list[dict]:
    """Per-provider totals over the window: turns, failures, chars, avg
    turn latency, and failure rate (failures / turns+failures). One row
    per provider with any activity in the window — the ops dashboard
    totals line. A provider that only failed still shows up, with
    failure_rate 1.0, so an outage reads as an outage, not as no usage.
    avg_fail_ms is the mean latency of that provider's FAILED attempts —
    kept separate from avg_ms, which covers only successful turns (a
    failover turn's wasted time belongs to the provider that wasted it,
    not to the one that eventually answered)."""
    c = _connect()
    days = max(1, days)
    cutoff = (datetime.date.today() - datetime.timedelta(days=days - 1)).isoformat()
    rows = c.execute(
        "SELECT provider, SUM(turns) AS turns, SUM(failures) AS failures,"
        " SUM(in_chars) AS in_chars, SUM(out_chars) AS out_chars,"
        " SUM(total_ms) AS total_ms, SUM(fail_ms) AS fail_ms"
        " FROM chat_stats WHERE day >= ? GROUP BY provider ORDER BY provider ASC",
        (cutoff,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        attempts = (d["turns"] or 0) + (d["failures"] or 0)
        d["avg_ms"] = round(d["total_ms"] / d["turns"]) if d["turns"] else 0
        d["avg_fail_ms"] = round(d["fail_ms"] / d["failures"]) if d["failures"] else 0
        d["failure_rate"] = round(d["failures"] / attempts, 4) if attempts else 0.0
        out.append(d)
    return out


def prune_stats(keep_days: int = 90) -> int:
    """Delete per-day rows older than keep_days (default 90). chat_stats
    grows one row per provider per active day forever, so without this
    the ops table is unbounded on Render's ephemeral disk. Returns the
    number of rows deleted. Called from the auto-reaper's housekeeping
    pass; keep_days comes from BRO_STATS_KEEP_DAYS there."""
    c = _connect()
    keep_days = max(1, keep_days)
    cutoff = (datetime.date.today() - datetime.timedelta(days=keep_days)).isoformat()
    with _lock:
        cur = c.execute("DELETE FROM chat_stats WHERE day < ?", (cutoff,))
        c.commit()
        return cur.rowcount
