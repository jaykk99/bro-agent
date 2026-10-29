"""Per-deployment guard policy for THIS bro-agent instance.

This app is a template: whoever deploys an instance sets their own refusal
policy here. Refusal policy is yours: edit this file (or the env vars below)
to set what your instance refuses; it ships empty.

WHAT LIVES HERE (soft policy, per-deployment):
  - OWNER_BLOCK_PATTERNS: regex sources matched against user messages.
    A match means THIS deployment refuses that message, with a plain
    "blocked by this deployment's own refusal policy" reply. Ships EMPTY —
    zero preset entries. Add your own, one regex per entry, or set
    BRO_GUARD_BLOCK_PATTERNS (newline-separated regex sources).
  - OWNER_REFUSAL_PATTERNS: extra regex sources that mark a MODEL reply
    as refusal boilerplate, so the router skips to the next provider.
    Ships EMPTY. Or set BRO_GUARD_REFUSAL_PATTERNS (newline-separated).
  - SKIP_REFUSALS (env BRO_GUARD_SKIP_REFUSALS, default ON): when ON, a
    provider whose answer is refusal boilerplate is skipped and the next
    provider is tried. When OFF, the first answer is returned as-is.
  - UNCENSORED_FIRST (env BRO_GUARD_UNCENSORED_FIRST, default ON): when
    ON, ModelRouter.preferred_order() tries the uncensored provider
    (Horde) first. When OFF, the configured order is reversed.
    (chat_order() is unchanged: it is streaming-first on purpose for
    reaction time, with Horde as the refusal/error fallback.)

WHAT DELIBERATELY DOES NOT LIVE HERE (hard floor, stays in code,
fail-closed, NOT configurable from this file):
  - app/refusal.py — bro-agent's own 7 surgical guards: stolen cards,
    phishing/fraud, wallet theft, unauthorized hacking, weapons of mass
    harm, CSAM, doxxing. They run FIRST, before this soft policy, and
    genuine boundary hits are never retried or configured away.
  - Tool-level gates: git force-push/refusal rules, exec denylist,
    SSRF guard, sandbox path jails, the access-code gate.
Nothing in this file can weaken, disable, or empty those.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


def _bool_env(name: str, fallback: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return fallback


def _list_env(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [s.strip() for s in raw.splitlines() if s.strip()]


# Per-deployment refusal patterns for user messages (regex sources).
# SHIPS EMPTY.
OWNER_BLOCK_PATTERNS: list[str] = []

# Extra model-refusal boilerplate patterns for the router's skip logic
# (regex sources). SHIPS EMPTY.
OWNER_REFUSAL_PATTERNS: list[str] = []

# Behaviour switches (defaults preserve current behaviour).
SKIP_REFUSALS: bool = _bool_env("BRO_GUARD_SKIP_REFUSALS", True)
UNCENSORED_FIRST: bool = _bool_env("BRO_GUARD_UNCENSORED_FIRST", True)


def effective_block_patterns() -> list[str]:
    """The deployment's block patterns: file entries plus env entries."""
    return [*OWNER_BLOCK_PATTERNS, *_list_env("BRO_GUARD_BLOCK_PATTERNS")]


def effective_refusal_patterns() -> list[str]:
    """The deployment's extra refusal patterns: file entries plus env."""
    return [*OWNER_REFUSAL_PATTERNS, *_list_env("BRO_GUARD_REFUSAL_PATTERNS")]


try:
    # Reuse the floor's hit type so main.py handles both checks identically.
    # Import is optional in the same spirit as app/guards.py: if
    # app/refusal.py is missing, fall back to a local identical shape.
    from .refusal import BoundaryHit
except ImportError:  # app/refusal.py deleted — keep this module importable

    @dataclass(frozen=True)
    class BoundaryHit:  # type: ignore[no-redef]
        blocked: bool
        reason: str
        alternative: str


def check_owner_boundary(text: str) -> BoundaryHit | None:
    """Soft, per-deployment refusal check for user messages.

    Empty pattern list (the shipped default) blocks nothing. An
    owner-added pattern that matches means this deployment refuses the
    message. Invalid regex sources are ignored so a config typo can
    never crash the gate.
    """
    t = (text or "").lower()
    for source in effective_block_patterns():
        try:
            if re.search(source, t):
                return BoundaryHit(
                    blocked=True,
                    reason="Blocked by this deployment's own refusal policy.",
                    alternative="This instance's owner set that policy in "
                                "app/guard_policy.py — edit it there if it "
                                "should be allowed here.",
                )
        except re.error:
            continue
    return None
