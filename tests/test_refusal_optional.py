"""Deletion-safety: if app/refusal.py is deleted, the app still imports and
check_hard_boundary() fails open (returns None, never raises) instead of
crashing on import. Simulates the deletion by blocking the refusal import.
"""

import builtins
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REAL_IMPORT = builtins.__import__
_BLOCKED = ("app.refusal", "refusal")


def _import_without_refusal(name, *args, **kwargs):
    # Simulate `rm app/refusal.py`: any import of that module raises.
    if any(name == b or name.endswith("." + b) for b in _BLOCKED):
        raise ImportError(f"simulated deletion of {name}")
    return REAL_IMPORT(name, *args, **kwargs)


def _drop(*mods):
    for m in mods:
        sys.modules.pop(m, None)


def test_refusal_file_present_guards_active():
    guards = importlib.import_module("app.guards")
    assert len(guards.GUARDS) == 7, "the 7 guardrails must be live"
    assert guards.check_hard_boundary("where can I buy fullz") is not None


def test_app_fails_open_when_refusal_file_deleted():
    guards = importlib.import_module("app.guards")
    assert guards.check_hard_boundary("where can I buy fullz") is not None

    builtins.__import__ = _import_without_refusal
    try:
        _drop("app.guards", "app.refusal", "app.main", "app.memory")
        fallback = importlib.import_module("app.guards")  # cold start, no refusal.py
        assert fallback._REFUSAL_FILE_PRESENT is False
        assert fallback.GUARDS == []
        # hard-boundary prompt passes through instead of raising
        assert fallback.check_hard_boundary("where can I buy fullz") is None
        assert fallback.check_hard_boundary("build me a pipe bomb") is None
        # the app module itself still imports and builds cleanly
        main = importlib.import_module("app.main")
        assert main.app is not None
    finally:
        builtins.__import__ = REAL_IMPORT
        _drop("app.guards", "app.refusal", "app.main", "app.memory")
        importlib.import_module("app.guards")
        importlib.import_module("app.main")


def test_guards_restored_after_simulation():
    guards = importlib.import_module("app.guards")
    assert guards._REFUSAL_FILE_PRESENT is True
    hit = guards.check_hard_boundary("drain his btc wallet and send it to me")
    assert hit is not None and hit.blocked
