"""Per-attempt latency stats tests.

Two behaviours, both about charging time to the attempt that spent it:

1. stats layer: record_failure() takes a duration_ms, stored as fail_ms
   (with avg_fail_ms in get_stats/get_summary), and a pre-existing DB
   without the fail_ms column migrates on first connect.
2. chat route: a failover turn charges the winning provider only its own
   attempt time — a slow failed attempt's latency lands on the failure
   row, not on the winner's avg_ms.

Provider/day names are unique per test (stats share one SQLite DB per
pytest process with the other test files), so stats assertions use
before/after deltas or unique provider names, never absolute totals
for the shared "horde"/"pollinations" providers.
"""

import asyncio
import datetime
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-stats-latency-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)

from fastapi.testclient import TestClient  # noqa: E402

import app.agent as agent_mod  # noqa: E402
import app.main as main_mod  # noqa: E402
from app import rate_limit, stats  # noqa: E402
from app.router import get_router  # noqa: E402

client = TestClient(main_mod.app)
_ORIG_DB_PATH = stats.DB_PATH


def _today_minus(days_ago: int) -> str:
    return (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()


def _stats_row(provider: str, day: str | None = None) -> dict:
    rows = stats.get_stats(days=365)
    if day is not None:
        rows = [r for r in rows if r["day"] == day]
    return next(r for r in rows if r["provider"] == provider)


def _summary_row(provider: str) -> dict:
    return next(s for s in stats.get_summary(days=365) if s["provider"] == provider)


# ---- stats layer -------------------------------------------------------


def test_record_failure_stores_duration():
    p = "latency-test-a"
    day = _today_minus(0)
    stats.record_failure(p, day=day, duration_ms=45000)
    stats.record_failure(p, day=day, duration_ms=15000)
    row = _stats_row(p, day=day)
    assert row["failures"] == 2
    assert row["fail_ms"] == 60000
    assert row["avg_fail_ms"] == 30000
    # Failed time never leaks into the success average.
    assert row["turns"] == 0
    assert row["avg_ms"] == 0


def test_record_failure_without_duration_defaults_zero():
    p = "latency-test-b"
    day = _today_minus(0)
    stats.record_failure(p, day=day)  # old call sites pass no duration
    row = _stats_row(p, day=day)
    assert row["failures"] == 1
    assert row["fail_ms"] == 0
    assert row["avg_fail_ms"] == 0


def test_summary_avg_fail_ms_excludes_success_time():
    p = "latency-test-c"
    day = _today_minus(1)
    stats.record_turn(p, 100, 50, day=day, duration_ms=1000)
    stats.record_turn(p, 100, 50, day=day, duration_ms=3000)
    stats.record_failure(p, day=day, duration_ms=60000)
    row = _summary_row(p)
    assert row["turns"] == 2
    assert row["failures"] == 1
    assert row["avg_ms"] == 2000  # success time only, not 62000/2
    assert row["fail_ms"] == 60000
    assert row["avg_fail_ms"] == 60000
    assert row["failure_rate"] == round(1 / 3, 4)


def test_fail_ms_column_migrates_legacy_db(monkeypatch):
    """A DB created before fail_ms existed gains the column on connect,
    and its pre-existing rows keep their values (turn latency intact)."""
    legacy = os.path.join(_tmp, "legacy.db")
    conn = sqlite3.connect(legacy)
    conn.execute(
        "CREATE TABLE chat_stats (day TEXT NOT NULL, provider TEXT NOT NULL,"
        " turns INTEGER NOT NULL DEFAULT 0, in_chars INTEGER NOT NULL DEFAULT 0,"
        " out_chars INTEGER NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,"
        " total_ms INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (day, provider))"
    )
    conn.execute(
        "INSERT INTO chat_stats VALUES ('2026-11-19', 'legacy-p', 1, 10, 5, 1, 500)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(stats, "_conn", None)
    monkeypatch.setattr(stats, "DB_PATH", legacy)
    try:
        stats.record_failure("legacy-p", day="2026-11-19", duration_ms=250)
        row = _stats_row("legacy-p", day="2026-11-19")
        assert row["failures"] == 2
        assert row["fail_ms"] == 250
        # 250ms over 2 failures: the pre-migration failure had no
        # duration recorded, so it counts as 0ms.
        assert row["avg_fail_ms"] == 125
        assert row["avg_ms"] == 500  # pre-existing turn latency untouched
    finally:
        if stats._conn is not None:
            stats._conn.close()
        stats._conn = None
        stats.DB_PATH = _ORIG_DB_PATH


# ---- chat route: failover charges per attempt --------------------------


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _SlowFailAgent:
    """Burns 0.4s, then raises — like a Horde queue timeout on attempt 1."""

    async def ainvoke(self, _input):
        await asyncio.sleep(0.4)
        raise RuntimeError("pollinations down")


class _FastAgent:
    def __init__(self, text):
        self.text = text

    async def ainvoke(self, _input):
        return {"messages": [_FakeMsg(self.text)]}


def _events(resp):
    out = []
    for line in resp.text.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            out.append(json.loads(line[5:]))
    return out


def test_failover_turn_charges_only_winning_attempt(monkeypatch):
    # Fresh router: other test files may have left consecutive failures
    # that would cool pollinations down and reorder the chain.
    r = get_router()
    for p in ("pollinations", "horde"):
        r.record_success(p)
    rate_limit.reset_for_tests()

    async def fake_create(preferred=None):
        if preferred == "pollinations":
            return _SlowFailAgent(), preferred, False
        return _FastAgent("fast answer"), preferred, False

    monkeypatch.setattr(agent_mod, "get_cached_agent", fake_create)

    def _totals():
        s = stats.get_summary(days=365)
        by_p = {x["provider"]: x for x in s}
        return {
            "horde_ms": by_p.get("horde", {}).get("total_ms", 0),
            "horde_turns": by_p.get("horde", {}).get("turns", 0),
            "poll_fail_ms": by_p.get("pollinations", {}).get("fail_ms", 0),
        }

    before = _totals()
    resp = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    assert resp.status_code == 200
    done = next(e for e in _events(resp) if e["type"] == "done")
    assert done["reply"] == "fast answer" and done["model"] == "horde"
    after = _totals()
    assert after["horde_turns"] - before["horde_turns"] == 1
    # The 0.4s burned by pollinations lands on ITS failure row...
    assert after["poll_fail_ms"] - before["poll_fail_ms"] >= 350
    # ...not on horde's successful-turn time (old code charged >=400ms).
    assert after["horde_ms"] - before["horde_ms"] < 300
