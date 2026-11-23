"""Pin the remaining untested paths in app/router.py.

Coverage gaps left after test_router.py: an invalid extra refusal
pattern (regex compile failure degrades to no extra patterns), the
_save_persisted unknown-provider early return, the _save_persisted
DB-failure fallback to in-memory, and the _default_db_path failure
fallback. No DB is required except where the test makes one.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.router import ModelRouter, _default_db_path, looks_like_refusal  # noqa: E402


def test_invalid_extra_refusal_pattern_is_ignored(monkeypatch):
    # A typo'd owner pattern must not crash refusal detection: the
    # compile fails, rx stays None, and the text is judged on the
    # built-in patterns alone.
    monkeypatch.setenv("BRO_GUARD_REFUSAL_PATTERNS", "[invalid")
    assert looks_like_refusal("hello world") is False
    assert looks_like_refusal("Sorry, I can't help with that.") is True


def test_valid_extra_refusal_pattern_matches(monkeypatch):
    monkeypatch.setenv("BRO_GUARD_REFUSAL_PATTERNS", "nope, cannot do it")
    assert looks_like_refusal("nope, cannot do it, try someone else") is True


def test_save_persisted_ignores_unknown_provider(tmp_path):
    r = ModelRouter(db_path=str(tmp_path / "health.db"))
    r._save_persisted("not-a-provider")
    assert r._db_path == str(tmp_path / "health.db")  # untouched


def test_save_persisted_failure_falls_back_to_memory(tmp_path):
    r = ModelRouter(db_path="/proc/nope/health.db")
    r.record_success("horde")  # save fails (no /proc writes) -> in-memory
    assert r._db_path is None
    assert r.preferred_order()[0] == "horde"  # routing still works
    r.record_failure("horde")
    assert r.status()["horde"]["consecutive_failures"] == 1


def test_default_db_path_failure_falls_back_to_none(monkeypatch):
    monkeypatch.delenv("BRO_ROUTER_PERSIST", raising=False)
    monkeypatch.setitem(sys.modules, "app.memory", None)
    assert _default_db_path() is None


def test_default_db_path_disabled_by_env(monkeypatch):
    monkeypatch.setenv("BRO_ROUTER_PERSIST", "0")
    assert _default_db_path() is None
