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

3. Cooldown persistence: provider health lives in the same SQLite DB as
   chat memory (table provider_health), so a cooldown survives the
   process restarts Render's free tier does on every sleep/wake. Without
   it, a provider that just failed twice got a clean slate on every wake
   and was retried first all over again. Persistence is best-effort: a
   DB error disables it for that router instance and routing carries on
   in memory, never raising. Disable with BRO_ROUTER_PERSIST=0.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field

from . import guard_policy

logger = logging.getLogger(__name__)

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
    """Picks provider order, tracks failures, applies cooldowns.

    Pass db_path to persist provider health across restarts (table
    provider_health in that SQLite file). db_path=None means in-memory
    only. The default (no db_path given) resolves to the chat-memory DB
    unless BRO_ROUTER_PERSIST=0.
    """

    _UNSET = object()

    def __init__(self, providers: tuple[str, ...] = PROVIDERS, db_path=_UNSET):
        self._states: dict[str, _ProviderState] = {p: _ProviderState(p) for p in providers}
        if db_path is self._UNSET:
            db_path = _default_db_path()
        self._db_path: str | None = db_path  # type: ignore[assignment]
        if self._db_path:
            self._load_persisted()

    def _load_persisted(self) -> None:
        """Restore health from the DB. A missing DB file means nothing was
        ever persisted — skip without creating anything. Any DB error
        disables persistence for this instance; routing stays in-memory."""
        assert self._db_path is not None
        try:
            if not os.path.exists(self._db_path):
                return
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS provider_health ("
                    "provider TEXT PRIMARY KEY, consecutive_failures INTEGER NOT NULL DEFAULT 0, "
                    "cooldown_until REAL NOT NULL DEFAULT 0)"
                )
                for provider, failures, cooldown_until in conn.execute(
                    "SELECT provider, consecutive_failures, cooldown_until FROM provider_health"
                ):
                    st = self._states.get(provider)
                    if st is not None:
                        st.consecutive_failures = max(0, int(failures or 0))
                        st.cooldown_until = float(cooldown_until or 0)
            finally:
                conn.close()
        except Exception:
            logger.warning("router health persistence unavailable, continuing in-memory", exc_info=True)
            self._db_path = None

    def _save_persisted(self, name: str) -> None:
        """Write one provider's health to the DB. Best-effort: a failure
        here must never break a chat turn, so it disables persistence
        for this instance and routing carries on in memory."""
        if not self._db_path:
            return
        st = self._states.get(name)
        if st is None:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS provider_health ("
                    "provider TEXT PRIMARY KEY, consecutive_failures INTEGER NOT NULL DEFAULT 0, "
                    "cooldown_until REAL NOT NULL DEFAULT 0)"
                )
                conn.execute(
                    "INSERT INTO provider_health (provider, consecutive_failures, cooldown_until)"
                    " VALUES (?, ?, ?) ON CONFLICT(provider) DO UPDATE SET"
                    " consecutive_failures=excluded.consecutive_failures,"
                    " cooldown_until=excluded.cooldown_until",
                    (name, st.consecutive_failures, st.cooldown_until),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.warning("router health save failed, continuing in-memory", exc_info=True)
            self._db_path = None

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
            self._save_persisted(name)

    def record_failure(self, name: str) -> None:
        st = self._states.get(name)
        if not st:
            return
        st.consecutive_failures += 1
        if st.consecutive_failures >= FAILURE_THRESHOLD:
            st.cooldown_until = time.time() + COOLDOWN_S
        self._save_persisted(name)

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
# not of one request. Built lazily so importing this module never touches
# the DB (tests that only need looks_like_refusal stay side-effect free).
_router: ModelRouter | None = None


def _default_db_path() -> str | None:
    """Where the process-wide router persists health: the chat-memory DB,
    unless BRO_ROUTER_PERSIST=0. A failure to resolve it means in-memory
    only — routing must never depend on the DB being reachable."""
    if os.environ.get("BRO_ROUTER_PERSIST", "1") == "0":
        return None
    try:
        from .memory import DB_PATH

        return DB_PATH
    except Exception:
        logger.warning("router health persistence unavailable, continuing in-memory", exc_info=True)
        return None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
