"""Per-client rate limiting for the chat endpoint — token bucket.

Single-user app, but the endpoint is internet-facing (Render) behind only
an access code, and each turn fans out to paid-ish/queue-based LLM
providers. The bucket caps bursts: 30 turns per minute sustained with a
burst of 8. Buckets are keyed by access code when the gate is on,
otherwise by client IP. Exceeding the limit raises RateLimited (the route
turns it into HTTP 429 with a Retry-After hint).
"""

from __future__ import annotations

import threading
import time

RATE_PER_SECOND = 30 / 60  # 30 turns/minute sustained
BURST = 8  # allow short bursts

_lock = threading.Lock()
_buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)


class RateLimited(Exception):
    def __init__(self, retry_after_s: float):
        super().__init__(f"rate limited, retry after {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s


def _refill(key: str, now: float) -> tuple[float, float]:
    tokens, last = _buckets.get(key, (BURST, now))
    tokens = min(BURST, tokens + (now - last) * RATE_PER_SECOND)
    return tokens, now


def check(key: str) -> None:
    """Consume one token for `key`. Raises RateLimited when the bucket is empty."""
    now = time.monotonic()
    with _lock:
        tokens, _ = _refill(key, now)
        if tokens < 1:
            # time until one token accrues
            retry_after = (1 - tokens) / RATE_PER_SECOND
            _buckets[key] = (tokens, now)
            raise RateLimited(retry_after)
        _buckets[key] = (tokens - 1, now)


def reset_for_tests() -> None:
    with _lock:
        _buckets.clear()
