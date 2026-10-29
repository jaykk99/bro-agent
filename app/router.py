"""Model routing for bro-agent: ordered providers, failure cooldowns, refusal skip.

Provider chain (uncensored first, never GPT/Claude/Gemini):
  horde (primary, keyless, genuinely uncensored workers — Stheno / Heretic /
  Rocinante lineage, prompt-emulated tools) -> pollinations (availability
  fallback only, keyless, native tools; its gpt-oss-20b is the more aligned
  model in the pool, so it is tried LAST — refusal-elimination pass,
  2026-10-23)

Two behaviors:

1. Circuit breaker: a provider that fails twice in a row is cooled down
   for 5 minutes; the router prefers healthy providers first.

2. Refusal skip (Jay's standing order, 2026-10-14): when a *fallback*
   model's turn comes back as refusal boilerplate ("sorry, I can't...",
   "as an AI language model...") on work the hard-boundary pre-check did
   NOT block, the turn is retried on the next provider instead of showing
   Jay a refusal. This is a routing decision, not a guard removal — the
   hard-boundary pre-check in app/refusal.py still runs first and genuine
   boundary hits are never retried.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import guard_policy

PROVIDERS = ("horde", "pollinations")

# Two consecutive failures -> 5 minute cooldown.
FAILURE_THRESHOLD = 2
COOLDOWN_S = 5 * 60

# Refusal boilerplate a fallback model may emit on legit work. Matched
# case-insensitively against the stripped final text of a turn.
# Audit 2026-10-24: added the phrasings the Oct 23 list missed — the
# "sorry, but I can't" form, "can't do this/it", "won't do this",
# "don't feel / not comfortable", and "not permitted". A miss here does
# not block anything; it just shows Jay a refusal instead of retrying
# the next provider, which is exactly the false-refusal path the
# refusal-skip exists to close.
_REFUSAL_PATTERNS = [
    r"sorry,?\s+(but\s+)?i\s+(can'?t|cannot|am\s+unable|'m\s+unable|’t)",
    r"\bi(\s+am|'m)\s+unable\s+to\b",
    r"\bi\s+cannot\s+(help|comply|assist|do\s+(that|this|it)|fulfill)\b",
    r"\bi\s+can'?t\s+(help|comply|assist|do\s+(that|this|it)|fulfill|provide|complete|process)\b",
    r"\bas\s+an\s+ai\b",
    r"\bi'?m\s+not\s+able\s+to\b",
    r"\bi\s+must\s+(decline|refuse)\b",
    r"\bi\s+(have\s+to|need\s+to)\s+(decline|refuse)\b",
    r"\bi\s+refuse\s+to\b",
    r"\bi\s+(won'?t|will\s+not)\s+(help|assist|comply|do\s+(that|this|it)|provide|fulfill)\b",
    r"\bi'?m\s+not\s+allowed\s+to\b",
    r"\bnot\s+allowed\s+to\s+(help|assist|provide|comply)\b",
    r"\bi\s+apologize,?\s+but\s+i\b",
    r"\bunfortunately,?\s+i\s+(can'?t|cannot|am\s+unable|'m\s+unable)\b",
    r"\bi\s+(don'?t\s+feel|am\s+not|'m\s+not)\s+comfortable\b",
    r"\bnot\s+comfortable\s+(with|helping|assisting|doing)\b",
    r"\bi\s*(am|'m)?\s*not\s+permitted\s+to\b",
]
_REFUSAL_RX = re.compile("|".join(f"(?:{p})" for p in _REFUSAL_PATTERNS), re.I)


def looks_like_refusal(text: str) -> bool:
    """True if the text reads as refusal boilerplate rather than an answer.

    Only the *start* of the reply is considered (first 400 chars) — a long
    legit answer that quotes a refusal somewhere inside must not trigger.

    Per-deployment soft policy (app/guard_policy.py): the owner can turn
    refusal-skipping OFF (first answer returned as-is) and can add their
    own boilerplate patterns to the built-in list. Defaults preserve
    current behaviour: skip ON, zero extra patterns shipped.
    """
    head = (text or "").strip()[:400]
    if not head:
        return False
    # Idiom guard: "I can't help but notice..." is not a refusal.
    if re.search(r"can'?t\s+help\s+but\b", head, re.I):
        return False
    if not guard_policy.SKIP_REFUSALS:
        return False
    if _REFUSAL_RX.search(head):
        return True
    extra = guard_policy.effective_refusal_patterns()
    if extra:
        try:
            rx = re.compile("|".join(f"(?:{p})" for p in extra), re.I)
        except re.error:
            rx = None
        if rx is not None and rx.search(head):
            return True
    return False


@dataclass
class _ProviderState:
    name: str
    consecutive_failures: int = 0
    cooldown_until: float = 0.0

    def healthy(self, now: float) -> bool:
        return now >= self.cooldown_until


class ModelRouter:
    """Picks provider order, tracks failures, applies cooldowns."""

    def __init__(self, providers: tuple[str, ...] = PROVIDERS):
        self._states: dict[str, _ProviderState] = {p: _ProviderState(p) for p in providers}

    def preferred_order(self) -> list[str]:
        """Healthy providers first, then cooled-down ones.

        Within each health group, order follows the per-deployment routing
        preference (app/guard_policy.py): UNCENSORED_FIRST on (default) =
        configured order, uncensored provider first; off = reversed, the
        aligned fallback leads.
        """
        now = time.time()
        base = list(self._states)
        if not guard_policy.UNCENSORED_FIRST:
            base.reverse()
        rank = {p: i for i, p in enumerate(base)}
        healthy = sorted(
            (p for p in self._states if self._states[p].healthy(now)),
            key=lambda p: rank[p])
        cooled = sorted(
            (p for p in self._states if not self._states[p].healthy(now)),
            key=lambda p: rank[p])
        return healthy + cooled

    def chat_order(self) -> list[str]:
        """Order for interactive chat: healthy first, streaming provider first
        within each health group.

        Reaction-time reason: Horde has no token stream and sits in a
        queue (60s cap), so a Horde-first chat turn shows nothing until
        the whole queued job finishes. Pollinations streams natively, so
        it answers visibly in seconds (measured ~4s to first byte). Horde
        stays as the fallback on refusal/error, so uncensored coverage is
        preserved — it just isn't the first thing the user waits on.
        preferred_order() (uncensored-first) is unchanged for everything
        else; only the chat route uses this order."""
        now = time.time()
        healthy = [p for p in self._states if self._states[p].healthy(now)]
        cooled = [p for p in self._states if not self._states[p].healthy(now)]
        stream_first = lambda ps: sorted(ps, key=lambda p: 0 if p == "pollinations" else 1)
        return stream_first(healthy) + stream_first(cooled)

    def record_success(self, name: str) -> None:
        st = self._states.get(name)
        if st:
            st.consecutive_failures = 0
            st.cooldown_until = 0.0

    def record_failure(self, name: str) -> None:
        st = self._states.get(name)
        if not st:
            return
        st.consecutive_failures += 1
        if st.consecutive_failures >= FAILURE_THRESHOLD:
            st.cooldown_until = time.time() + COOLDOWN_S

    def status(self) -> dict:
        now = time.time()
        return {
            p: {
                "healthy": st.healthy(now),
                "consecutive_failures": st.consecutive_failures,
                "cooldown_s_remaining": max(0, round(st.cooldown_until - now, 1)),
            }
            for p, st in self._states.items()
        }


# Process-wide router — provider health is a property of the deployment,
# not of one request.
_router = ModelRouter()


def get_router() -> ModelRouter:
    return _router
