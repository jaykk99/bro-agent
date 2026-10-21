"""Context windowing for chat history.

The old chat route fed the last 60 messages to the model with no size cap —
a long thread could blow past the model's context window (Horde workers get
8k) and waste the queue wait on a prompt that gets truncated mid-thought.

build_window() keeps the most recent messages that fit inside a character
budget and, when anything is dropped, inserts a marker noting how many
older messages were trimmed so the model knows context is missing instead
of hallucinating continuity.
"""

from __future__ import annotations

DEFAULT_MAX_CHARS = 24_000  # ~8k tokens at 3 chars/token


def build_window(history: list[dict], max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """Newest-first fit inside max_chars; chronological result.

    Always keeps at least the single newest message. Dropped messages are
    summarized by a leading marker message.
    """
    if not history:
        return []
    budget = max_chars
    kept: list[dict] = []
    dropped = 0
    for msg in reversed(history):
        content = str(msg.get("content", ""))
        cost = len(content) + 16  # role overhead
        if kept and cost > budget:
            dropped += 1
            continue
        kept.append(msg)
        budget -= cost
    kept.reverse()
    if dropped:
        kept.insert(
            0,
            {
                "role": "system",
                "content": f"[{dropped} older message(s) trimmed to fit context; continue from what follows]",
            },
        )
    return kept


def estimate_chars(history: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in history)
