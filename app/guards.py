"""Compatibility shim — all refusal code moved to app/refusal.py (owner order,
Jay, 2026-10-13): the 7 guardrails and the boundary check that enforces them
live THERE and only there. Existing `from .guards import ...` calls keep
working unchanged.

Deletion-safe by design: if app/refusal.py is deleted, this module fails
open — GUARDS is empty, check_hard_boundary() returns None, and the app
runs without the boundary pre-check instead of crashing on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

try:
    from .refusal import GUARDS, BoundaryHit, Guard, check_hard_boundary  # noqa: F401

    _REFUSAL_FILE_PRESENT = True
except ImportError:  # app/refusal.py deleted — fail open, the app stays up
    _REFUSAL_FILE_PRESENT = False

    @dataclass(frozen=True)
    class BoundaryHit:  # type: ignore[no-redef]
        blocked: bool
        reason: str
        alternative: str

    @dataclass(frozen=True)
    class Guard:  # type: ignore[no-redef]
        id: str
        match: Callable[[str], bool]
        reason: str
        alternative: str

    GUARDS: List[Guard] = []

    _warned = False

    def check_hard_boundary(text: str) -> BoundaryHit | None:
        """No refusal file → no pre-check. Everything passes through to the model."""
        global _warned
        if not _warned:
            print(
                "[guards] app/refusal.py is missing — running WITHOUT the "
                "hard-boundary pre-check (fail open, app stays up)",
                flush=True,
            )
            _warned = True
        return None
