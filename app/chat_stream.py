"""Real token streaming for POST /api/chat (chat-speed fix 2).

Before this, /api/chat was SSE in name only: it ran the whole turn with
agent.ainvoke() and then emitted one `done` event, so the user saw
nothing until the entire reply (possibly several model + tool rounds)
finished.

This module drives one agent turn and emits token deltas as they arrive:

  * Preferred path — agent.astream_events (what UniHarness's Agent
    exposes, wrapping LangGraph's astream_events v2): `on_chat_model_stream`
    events carry per-token AIMessageChunk content for models that stream
    natively (Pollinations via ChatOpenAI). Each chunk's text is emitted
    as {"type": "token", "delta": ...} on the existing SSE stream.
  * Fallback path — agent.astream (graph state stream): node-level state,
    not tokens, so no token events are faked from it; the final message
    is still extracted.
  * Fallback path — agent.ainvoke only: same as the old behaviour
    (one reply, no tokens). No synthetic token events are emitted.

Final-reply authority: the `done` event still carries the turn's final
message, exactly as before. The final text is the LAST on_chat_model_end
output (the model's final turn, after any tool rounds), NOT the concat
of streamed deltas — intermediate turns (planning narration before a
tool call) stream live but do not become the answer. If no end event
arrives, the accumulated deltas are the fallback final text. The
frontend replaces the streamed preview with the `done` reply, so a
provider hop (refusal/error failover) never leaves a stale preview
as the answer.

Horde honesty note: HordeChatModel has no token stream — its _generate
returns only when the Horde job completes — so LangChain synthesizes
its stream as a single chunk at job completion. A Horde turn therefore
emits at most one token event (the whole reply) just before `done`,
and its real speed win is the 60s cap + provider failover, not tokens.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

Emit = Callable[[dict], Awaitable[None]]


def message_text(msg: Any) -> str:
    """Best-effort text of a LangChain message / chunk / result."""
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    # ChatResult-like: generations[0].message
    gens = getattr(msg, "generations", None)
    if gens:
        return message_text(gens[0].message) if gens else ""
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    if content is not None:
        return str(content)
    return ""


def _event_field(event: Any, name: str) -> Any:
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def stream_delta(event: Any) -> str | None:
    """Token text from an astream_events v2 event, else None."""
    if _event_field(event, "event") != "on_chat_model_stream":
        return None
    data = _event_field(event, "data") or {}
    chunk = data.get("chunk") if isinstance(data, dict) else getattr(data, "chunk", None)
    text = message_text(chunk)
    return text if text else None


def end_text(event: Any) -> str | None:
    """Final text of a model turn from an on_chat_model_end event, else None."""
    if _event_field(event, "event") != "on_chat_model_end":
        return None
    data = _event_field(event, "data") or {}
    output = data.get("output") if isinstance(data, dict) else getattr(data, "output", None)
    text = message_text(output)
    return text if text else None


def _final_from_state(state: Any) -> str:
    """Final message text from a graph state dict (astream fallback)."""
    if isinstance(state, dict):
        for key in ("messages",):
            msgs = state.get(key)
            if msgs:
                return message_text(msgs[-1])
        for value in state.values():
            if isinstance(value, dict):
                text = _final_from_state(value)
                if text:
                    return text
    return ""


async def stream_agent_turn(agent: Any, messages: list[dict], emit: Emit) -> str:
    """Run one agent turn, emitting token events via `emit`, return final text.

    `messages` is the [{"role": ..., "content": ...}] list the chat route
    already builds. The caller wraps this in its turn timeout.
    """
    payload = {"messages": messages}
    if hasattr(agent, "astream_events"):
        accumulated: list[str] = []
        last_end = ""
        async for event in agent.astream_events(payload):
            delta = stream_delta(event)
            if delta is not None:
                accumulated.append(delta)
                await emit({"type": "token", "delta": delta})
                continue
            text = end_text(event)
            if text is not None:
                last_end = text
        return last_end or "".join(accumulated)
    if hasattr(agent, "astream"):
        final = ""
        async for state in agent.astream(payload):
            text = _final_from_state(state)
            if text:
                final = text
        return final
    result = await agent.ainvoke(payload)
    out_messages = result["messages"] if isinstance(result, dict) else []
    if out_messages:
        return message_text(out_messages[-1])
    return ""
