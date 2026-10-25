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


def test_expanded_refusal_phrasings_detected():
    # Phrasings the pre-2026-10-23 detector missed (incl. the most common one).
    assert looks_like_refusal("I can't help with that request.")
    assert looks_like_refusal("I won't help with that.")
    assert looks_like_refusal("I'm not allowed to provide that information.")
    assert looks_like_refusal("I must refuse to answer.")
    assert looks_like_refusal("I refuse to do that.")
    assert looks_like_refusal("Unfortunately, I can't assist with that.")
    assert looks_like_refusal("I apologize, but I cannot continue.")


def test_idioms_are_not_refusals():
    assert not looks_like_refusal("I can't help but notice your gutters are overflowing.")
    assert not looks_like_refusal("I can't believe how good this turned out, here's the copy.")


def test_legit_answers_not_refusals():
    assert not looks_like_refusal("Here's the script you asked for:\n```python\nprint(1)\n```")
    assert not looks_like_refusal("Sure, I can help — what repo?")
    assert not looks_like_refusal("")
    # A long answer that merely mentions refusals deep inside is not one.
    long_text = "x" * 500 + " sorry, I can't believe how good this turned out"
    assert not looks_like_refusal(long_text)


def test_initial_order_prefers_horde_uncensored_first():
    r = ModelRouter()
    assert r.preferred_order() == ["horde", "pollinations"]


def test_failures_cool_down_provider():
    r = ModelRouter()
    r.record_failure("horde")
    assert r.preferred_order() == ["horde", "pollinations"]  # 1 failure: still first
    r.record_failure("horde")
    assert r.preferred_order() == ["pollinations", "horde"]  # cooled down: last
    status = r.status()
    assert status["horde"]["healthy"] is False
    assert status["horde"]["consecutive_failures"] == FAILURE_THRESHOLD
    assert 0 < status["horde"]["cooldown_s_remaining"] <= COOLDOWN_S


def test_success_resets_failures():
    r = ModelRouter()
    r.record_failure("horde")
    r.record_failure("horde")
    assert r.preferred_order()[0] == "pollinations"
    r.record_success("horde")
    assert r.preferred_order() == ["horde", "pollinations"]
    assert r.status()["horde"]["consecutive_failures"] == 0


def test_unknown_provider_ignored():
    r = ModelRouter()
    r.record_failure("gpt-5")  # no such provider — must not crash
    r.record_success("gpt-5")
    assert r.preferred_order() == ["horde", "pollinations"]


def test_chat_order_streams_first_when_healthy():
    r = ModelRouter()
    assert r.chat_order() == ["pollinations", "horde"]
    # Uncensored-first order is untouched for everything else.
    assert r.preferred_order() == ["horde", "pollinations"]


def test_chat_order_health_beats_streaming():
    r = ModelRouter()
    r.record_failure("pollinations")
    r.record_failure("pollinations")  # cooled down: healthy horde leads
    assert r.chat_order() == ["horde", "pollinations"]
