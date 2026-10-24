"""Tests for rate limiting and context windowing."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import rate_limit  # noqa: E402
from app.context import build_window, estimate_chars  # noqa: E402
from app.rate_limit import BURST, RateLimited  # noqa: E402


def test_rate_limit_allows_burst_then_blocks():
    rate_limit.reset_for_tests()
    for _ in range(BURST):
        rate_limit.check("k1")
    try:
        rate_limit.check("k1")
    except RateLimited as e:
        assert e.retry_after_s > 0
    else:
        raise AssertionError("expected RateLimited")


def test_rate_limit_keys_independent():
    rate_limit.reset_for_tests()
    for _ in range(BURST):
        rate_limit.check("a")
    rate_limit.check("b")  # different key unaffected


def test_rate_limit_refills(monkeypatch):
    rate_limit.reset_for_tests()
    now = [1000.0]
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: now[0])
    for _ in range(BURST):
        rate_limit.check("r")
    now[0] += 130  # >2 min: full refill (30/min)
    for _ in range(BURST):
        rate_limit.check("r")


def test_window_keeps_recent_within_budget():
    hist = [{"role": "user", "content": f"msg{i} " + "x" * 100} for i in range(20)]
    win = build_window(hist, max_chars=500)
    assert win
    assert win[-1]["content"].startswith("msg19")
    # chronological order preserved
    contents = [m["content"] for m in win if m["role"] != "system"]
    assert contents == sorted(contents, key=lambda c: int(c.split()[0][3:]))


def test_window_inserts_trim_marker():
    hist = [{"role": "user", "content": "x" * 1000} for _ in range(10)]
    win = build_window(hist, max_chars=2500)
    assert win[0]["role"] == "system"
    assert "trimmed" in win[0]["content"]


def test_window_no_trim_no_marker():
    hist = [{"role": "user", "content": "hi"}]
    win = build_window(hist, max_chars=2500)
    assert win == hist


def test_window_empty():
    assert build_window([]) == []


def test_estimate_chars():
    assert estimate_chars([{"content": "abc"}, {"content": "de"}]) == 5
