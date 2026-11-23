"""Pin the remaining untested paths in app/reaper.py.

Coverage gaps left after test_reaper.py: the forge half failing
while the train half is fine (first_error records the forge error),
a garbage BRO_STATS_KEEP_DAYS falling back to 90, and a stats-prune
failure never marking the reap pass as failed.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-reaper-gaps-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

from app import dispatch, reaper, stats  # noqa: E402


def test_run_reap_once_records_forge_half_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("forge db down")

    monkeypatch.setattr(dispatch, "reap_dead_workers", _boom)
    out = reaper.run_reap_once()
    assert set(out) == {"requeued", "failed", "reaped_workers"}
    assert reaper.status()["last_error"] == "forge reap failed: forge db down"


def test_run_reap_once_bad_keep_days_falls_back_to_90(monkeypatch):
    monkeypatch.setenv("BRO_STATS_KEEP_DAYS", "not-a-number")
    seen = {}
    monkeypatch.setattr(stats, "prune_stats", lambda days: seen.update(days=days))
    reaper.run_reap_once()
    assert seen["days"] == 90


def test_run_reap_once_prune_failure_does_not_fail_pass(monkeypatch):
    def _boom(days):
        raise RuntimeError("stats db down")

    monkeypatch.setattr(stats, "prune_stats", _boom)
    out = reaper.run_reap_once()
    assert set(out) == {"requeued", "failed", "reaped_workers"}
    assert reaper.status()["last_error"] is None
