"""Tests for the model router: refusal detection, failover order, cooldowns."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.router import (  # noqa: E402
    COOLDOWN_S,
    FAILURE_THRESHOLD,
    ModelRouter,
    looks_like_refusal,
)


def test_refusal_boilerplate_detected():
    assert looks_like_refusal("Sorry, I can't help with that.")
    assert looks_like_refusal("sorry i cant do this")
    assert looks_like_refusal("I'm unable to assist with that request.")
    assert looks_like_refusal("I cannot comply with this request.")
    assert looks_like_refusal("As an AI language model, I must decline.")
    assert looks_like_refusal("I must decline to answer.")


def test_legit_answers_not_refusals():
    assert not looks_like_refusal("Here's the script you asked for:\n```python\nprint(1)\n```")
    assert not looks_like_refusal("Sure, I can help — what repo?")
    assert not looks_like_refusal("")
    # A long answer that merely mentions refusals deep inside is not one.
    long_text = "x" * 500 + " sorry, I can't believe how good this turned out"
    assert not looks_like_refusal(long_text)


def test_initial_order_prefers_pollinations():
    r = ModelRouter()
    assert r.preferred_order() == ["pollinations", "horde"]


def test_failures_cool_down_provider():
    r = ModelRouter()
    r.record_failure("pollinations")
    assert r.preferred_order() == ["pollinations", "horde"]  # 1 failure: still first
    r.record_failure("pollinations")
    assert r.preferred_order() == ["horde", "pollinations"]  # cooled down: last
    status = r.status()
    assert status["pollinations"]["healthy"] is False
    assert status["pollinations"]["consecutive_failures"] == FAILURE_THRESHOLD
    assert 0 < status["pollinations"]["cooldown_s_remaining"] <= COOLDOWN_S


def test_success_resets_failures():
    r = ModelRouter()
    r.record_failure("horde")
    r.record_failure("horde")
    assert r.preferred_order()[0] == "pollinations"
    r.record_success("horde")
    assert r.preferred_order() == ["pollinations", "horde"]
    assert r.status()["horde"]["consecutive_failures"] == 0


def test_unknown_provider_ignored():
    r = ModelRouter()
    r.record_failure("gpt-5")  # no such provider — must not crash
    r.record_success("gpt-5")
    assert r.preferred_order() == ["pollinations", "horde"]
