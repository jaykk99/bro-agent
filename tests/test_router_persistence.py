"""Tests for router cooldown persistence (app/router.py).

Provider health survives a process restart via the provider_health table,
so a Render sleep/wake does not give a just-failed provider a clean slate.
All tests use an explicit throwaway DB path — never the real chat DB.
"""

import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Bind the chat DB to a throwaway file before importing app modules:
# app.memory reads BRO_DB_PATH once at import time, and a late env set
# would silently point the global router at the repo's real data/bro.db.
_tmp = tempfile.mkdtemp(prefix="bro-router-persist-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

import app.memory as memory_mod  # noqa: E402
import app.router as router_mod  # noqa: E402
from app.router import COOLDOWN_S, FAILURE_THRESHOLD, ModelRouter, get_router  # noqa: E402


def _tmp_db() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="bro-router-persist-"), "test.db")


def test_cooldown_survives_restart():
    path = _tmp_db()
    r1 = ModelRouter(db_path=path)
    r1.record_failure("horde")
    r1.record_failure("horde")
    assert r1.status()["horde"]["healthy"] is False
    # A fresh router on the same DB is a simulated restart: the cooldown
    # must still be there, and the cooled-down provider sorts last.
    r2 = ModelRouter(db_path=path)
    assert r2.status()["horde"]["healthy"] is False
    assert r2.status()["horde"]["consecutive_failures"] == FAILURE_THRESHOLD
    assert r2.preferred_order() == ["pollinations", "horde"]


def test_success_clears_persisted_cooldown():
    path = _tmp_db()
    r1 = ModelRouter(db_path=path)
    r1.record_failure("horde")
    r1.record_failure("horde")
    r2 = ModelRouter(db_path=path)
    assert r2.status()["horde"]["healthy"] is False
    r2.record_success("horde")
    r3 = ModelRouter(db_path=path)
    assert r3.status()["horde"]["healthy"] is True
    assert r3.status()["horde"]["consecutive_failures"] == 0
    assert r3.preferred_order() == ["horde", "pollinations"]


def test_expired_cooldown_loads_healthy():
    path = _tmp_db()
    r1 = ModelRouter(db_path=path)
    r1.record_failure("horde")
    r1.record_failure("horde")
    # Backdate the stored cooldown past expiry, as if the 5 minutes elapsed
    # while the process was down.
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE provider_health SET cooldown_until=? WHERE provider=?",
        (time.time() - 10, "horde"),
    )
    conn.commit()
    conn.close()
    r2 = ModelRouter(db_path=path)
    assert r2.status()["horde"]["healthy"] is True
    assert r2.status()["horde"]["consecutive_failures"] == FAILURE_THRESHOLD


def test_unknown_provider_rows_ignored():
    path = _tmp_db()
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE provider_health (provider TEXT PRIMARY KEY,"
        " consecutive_failures INTEGER NOT NULL DEFAULT 0,"
        " cooldown_until REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO provider_health VALUES ('gpt-5', 99, ?)",
        (time.time() + COOLDOWN_S,),
    )
    conn.commit()
    conn.close()
    r = ModelRouter(db_path=path)
    assert sorted(r.status()) == ["horde", "pollinations"]
    assert r.preferred_order() == ["horde", "pollinations"]


def test_missing_db_file_loads_clean_without_creating_it():
    path = os.path.join(tempfile.mkdtemp(prefix="bro-router-persist-"), "nope.db")
    r = ModelRouter(db_path=path)
    assert r.preferred_order() == ["horde", "pollinations"]
    assert not os.path.exists(path)  # a read-only load must not create the DB


def test_in_memory_router_creates_no_file():
    path = _tmp_db()
    r = ModelRouter(db_path=None)
    r.record_failure("horde")
    r.record_failure("horde")
    assert r.status()["horde"]["healthy"] is False
    assert not os.path.exists(path)


def test_unwritable_db_never_breaks_routing():
    # Pointing persistence at a directory: every DB op fails, and routing
    # must carry on in memory instead of raising into a chat turn.
    bad = tempfile.mkdtemp(prefix="bro-router-persist-")
    r = ModelRouter(db_path=bad)
    r.record_failure("horde")
    r.record_failure("horde")
    assert r.status()["horde"]["healthy"] is False
    assert r.preferred_order() == ["pollinations", "horde"]
    r.record_success("horde")
    assert r.status()["horde"]["healthy"] is True


def test_get_router_persists_across_rebuilds(monkeypatch):
    path = _tmp_db()
    monkeypatch.setattr(memory_mod, "DB_PATH", path)
    monkeypatch.setattr(router_mod, "_router", None)
    try:
        r1 = get_router()
        assert get_router() is r1  # same process-wide instance
        r1.record_failure("pollinations")
        r1.record_failure("pollinations")
        assert r1.status()["pollinations"]["healthy"] is False
        monkeypatch.setattr(router_mod, "_router", None)
        r2 = get_router()
        assert r2 is not r1
        assert r2.status()["pollinations"]["healthy"] is False
        assert r2.status()["pollinations"]["consecutive_failures"] == FAILURE_THRESHOLD
    finally:
        monkeypatch.setattr(router_mod, "_router", None)


def test_persist_disabled_by_env(monkeypatch):
    path = _tmp_db()
    monkeypatch.setattr(memory_mod, "DB_PATH", path)
    monkeypatch.setenv("BRO_ROUTER_PERSIST", "0")
    monkeypatch.setattr(router_mod, "_router", None)
    try:
        r = get_router()
        r.record_failure("horde")
        r.record_failure("horde")
        assert r.status()["horde"]["healthy"] is False  # in-memory still works
        assert not os.path.exists(path)  # but nothing was persisted
    finally:
        monkeypatch.setattr(router_mod, "_router", None)
