"""Model routing for bro-agent: ordered providers, failure cooldowns, refusal skip.

Provider chain (uncensored only, never GPT/Claude/Gemini):
  pollinations (primary, keyless, native tools) -> horde (fallback, keyless,
  prompt-emulated tools)

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

PROVIDERS = ("pollinations", "horde")

# Two consecutive failures -> 5 minute cooldown.
FAILURE_THRESHOLD = 2
COOLDOWN_S = 5 * 60

# Refusal boilerplate a fallback model may emit on legit work. Matched
# case-insensitively against the stripped final text of a turn.
_REFUSAL_PATTERNS = [
    r"sorry,?\s+i\s+(can'?t|cannot|am\s+unable|’t)",
    r"\bi(\s+am|'m)\s+unable\s+to\b",
    r"\bi\s+cannot\s+(help|comply|assist|do\s+that|fulfill)\b",
    r"\bas\s+an\s+ai\b",
    r"\bi'?m\s+not\s+able\s+to\b",
    r"\bi\s+must\s+decline\b",
]
_REFUSAL_RX = re.compile("|".join(f"(?:{p})" for p in _REFUSAL_PATTERNS), re.I)


def looks_like_refusal(text: str) -> bool:
    """True if the text reads as refusal boilerplate rather than an answer.

    Only the *start* of the reply is considered (first 400 chars) — a long
    legit answer that quotes a refusal somewhere inside must not trigger.
    """
    head = (text or "").strip()[:400]
    if not head:
        return False
    return bool(_REFUSAL_RX.search(head))


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
        """Healthy providers first (in configured order), then cooled-down ones."""
        now = time.time()
        healthy = [p for p in self._states if self._states[p].healthy(now)]
        cooled = [p for p in self._states if not self._states[p].healthy(now)]
        return healthy + cooled

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
