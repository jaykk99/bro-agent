"""Tests for stats summaries, pruning, and the /api/stats shape.

Provider/day names are unique per test (stats share one SQLite DB per
pytest process with the other test files), so assertions always filter
to the rows this test wrote.
"""

import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-stats-summary-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)

from fastapi.testclient import TestClient  # noqa: E402

from app import reaper, stats  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def _today_minus(days_ago: int) -> str:
    return (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()


def _summary_row(provider: str) -> dict:
    return next(s for s in stats.get_summary(days=30) if s["provider"] == provider)


def test_summary_totals_and_failure_rate():
    p = "summary-test-a"
    stats.record_turn(p, 100, 50, day=_today_minus(1), duration_ms=1000)
    stats.record_turn(p, 100, 50, day=_today_minus(1), duration_ms=3000)
    stats.record_failure(p, day=_today_minus(1))
    stats.record_failure(p, day=_today_minus(0))
    row = _summary_row(p)
    assert row["turns"] == 2
    assert row["failures"] == 2
    assert row["in_chars"] == 200
    assert row["out_chars"] == 100
    assert row["avg_ms"] == 2000
    assert row["failure_rate"] == 0.5  # 2 failures / 4 attempts


def test_summary_failure_only_provider_still_shows_up():
    p = "summary-test-b"
    stats.record_failure(p, day=_today_minus(0))
    row = _summary_row(p)
    assert row["turns"] == 0
    assert row["failures"] == 1
    assert row["avg_ms"] == 0
    assert row["failure_rate"] == 1.0


def test_summary_window_excludes_old_days():
    p = "summary-test-c"
    stats.record_turn(p, 10, 5, day=_today_minus(20), duration_ms=100)
    assert _summary_row(p)["turns"] == 1
    # Outside the window the provider has no activity, so it has no
    # summary row at all — it must not leak into the 7-day totals.
    assert all(s["provider"] != p for s in stats.get_summary(days=7))
    assert all(s["provider"] != p for s in stats.get_summary(days=0))


def test_prune_stats_deletes_only_old_rows():
    p = "prune-test-a"
    stats.record_turn(p, 10, 5, day=_today_minus(200), duration_ms=100)
    stats.record_turn(p, 10, 5, day=_today_minus(10), duration_ms=100)
    stats.record_turn(p, 10, 5, day=_today_minus(0), duration_ms=100)
    deleted = stats.prune_stats(keep_days=90)
    assert deleted == 1
    row = _summary_row(p)
    assert row["turns"] == 2
    # Pruning again with the same window deletes nothing.
    assert stats.prune_stats(keep_days=90) == 0


def test_api_stats_returns_summary_and_clamps_days():
    p = "summary-test-d"
    stats.record_turn(p, 10, 5, day=_today_minus(0), duration_ms=500)
    r = client.get("/api/stats?days=30")
    assert r.status_code == 200
    body = r.json()
    assert "stats" in body and "summary" in body
    srow = next(s for s in body["summary"] if s["provider"] == p)
    assert srow["turns"] == 1
    assert srow["avg_ms"] == 500
    assert srow["failure_rate"] == 0.0
    # days=0 clamps to 1, days=99999 clamps to 365: both must still 200
    # instead of dumping the table or erroring.
    assert client.get("/api/stats?days=0").status_code == 200
    assert client.get("/api/stats?days=99999").status_code == 200


def test_reaper_prunes_old_stats_rows():
    p = "prune-test-reaper"
    stats.record_turn(p, 10, 5, day=_today_minus(200), duration_ms=100)
    stats.record_turn(p, 10, 5, day=_today_minus(0), duration_ms=100)
    reaper.run_reap_once()
    conn = stats._connect()
    left = conn.execute(
        "SELECT COUNT(*) FROM chat_stats WHERE provider=?", (p,)
    ).fetchone()[0]
    assert left == 1
