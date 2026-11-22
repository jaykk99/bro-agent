"""Chat streaming helpers: message text extraction, event fields,
state fallback, and all three stream_agent_turn paths.

No network, no real LLM: fake messages, events, and agents only.
Covers the branches test_chat_speed.py leaves open (message_text
content shapes, _final_from_state, astream / ainvoke fallbacks).
"""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.chat_stream import (  # noqa: E402
    _event_field,
    _final_from_state,
    end_text,
    message_text,
    stream_agent_turn,
    stream_delta,
)


class FakeMsg:
    def __init__(self, content):
        self.content = content


class TextBlock:
    """A content block exposing .text but no dict shape."""

    def __init__(self, text):
        self.text = text


class FakeResult:
    """ChatResult-like: generations[0].message."""

    def __init__(self, message):
        self.generations = [types.SimpleNamespace(message=message)]


# ---- message_text ---------------------------------------------------------

def test_message_text_none_and_str():
    assert message_text(None) == ""
    assert message_text("plain") == "plain"


def test_message_text_message_object_str_content():
    assert message_text(FakeMsg("hello")) == "hello"


def test_message_text_chat_result_unwraps_first_generation():
    assert message_text(FakeResult(FakeMsg("gen text"))) == "gen text"


def test_message_text_empty_generations_falls_through():
    result = FakeResult(FakeMsg("unused"))
    result.generations = []
    assert message_text(result) == ""


def test_message_text_dict_content():
    assert message_text({"content": "dict text"}) == "dict text"
    assert message_text({"content": None}) == ""
    assert message_text({}) == ""


def test_message_text_list_blocks_mixed_shapes():
    msg = FakeMsg([
        "lead ",
        {"type": "text", "text": "dict "},
        {"type": "image_url", "image_url": "http://x/y.png"},
        TextBlock("attr "),
        {"type": "text"},
    ])
    assert message_text(msg) == "lead dict attr "


def test_message_text_list_blocks_empty():
    assert message_text(FakeMsg([])) == ""


def test_message_text_non_str_content_stringified():
    assert message_text(FakeMsg(42)) == "42"


def test_message_text_no_content_attr():
    assert message_text(object()) == ""


# ---- _event_field / stream_delta / end_text -------------------------------

def test_event_field_dict_and_object():
    assert _event_field({"event": "x"}, "event") == "x"
    assert _event_field(types.SimpleNamespace(event="y"), "event") == "y"
    assert _event_field(types.SimpleNamespace(), "event") is None
    assert _event_field({"event": "x"}, "missing") is None


def test_stream_delta_object_style_event():
    event = types.SimpleNamespace(
        event="on_chat_model_stream",
        data=types.SimpleNamespace(chunk=FakeMsg("tok")),
    )
    assert stream_delta(event) == "tok"


def test_stream_delta_empty_chunk_is_none():
    assert stream_delta(
        {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg("")}}
    ) is None
    assert stream_delta({"event": "on_chat_model_stream", "data": {}}) is None
    assert stream_delta({"event": "on_chat_model_stream"}) is None


def test_stream_delta_wrong_event_is_none():
    assert stream_delta(
        {"event": "on_chat_model_end", "data": {"output": FakeMsg("x")}}
    ) is None


def test_end_text_object_style_event_and_empty():
    event = types.SimpleNamespace(
        event="on_chat_model_end",
        data=types.SimpleNamespace(output=FakeMsg("final")),
    )
    assert end_text(event) == "final"
    assert end_text({"event": "on_chat_model_end", "data": {"output": FakeMsg("")}}) is None
    assert end_text({"event": "on_chat_model_end", "data": {}}) is None
    assert end_text({"event": "on_chat_model_end"}) is None


# ---- _final_from_state -----------------------------------------------------

def test_final_from_state_messages_last():
    state = {"messages": [FakeMsg("first"), FakeMsg("last")]}
    assert _final_from_state(state) == "last"


def test_final_from_state_empty_messages_falls_to_nested():
    state = {"messages": [], "agent": {"messages": [FakeMsg("nested")]}}
    assert _final_from_state(state) == "nested"


def test_final_from_state_no_messages_anywhere():
    assert _final_from_state({"other": "x"}) == ""
    assert _final_from_state({"other": {"deep": 1}}) == ""
    assert _final_from_state(None) == ""
    assert _final_from_state("not a dict") == ""


# ---- stream_agent_turn: astream_events path -------------------------------

def test_stream_turn_no_end_event_uses_accumulated():
    emitted = []

    async def emit(evt):
        emitted.append(evt)

    class StreamOnlyAgent:
        async def astream_events(self, _input):
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg("a")}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg("b")}}
            yield {"event": "on_tool_start", "data": {}}

    final = asyncio.run(stream_agent_turn(StreamOnlyAgent(), [], emit))
    assert final == "ab"
    assert [e["delta"] for e in emitted] == ["a", "b"]


def test_stream_turn_empty_end_event_uses_accumulated():
    emitted = []

    async def emit(evt):
        emitted.append(evt)

    class EmptyEndAgent:
        async def astream_events(self, _input):
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg("x")}}
            yield {"event": "on_chat_model_end", "data": {"output": FakeMsg("")}}

    final = asyncio.run(stream_agent_turn(EmptyEndAgent(), [], emit))
    assert final == "x"


def test_stream_turn_no_events_returns_empty():
    emitted = []

    async def emit(evt):
        emitted.append(evt)

    class SilentAgent:
        async def astream_events(self, _input):
            if False:
                yield None

    assert asyncio.run(stream_agent_turn(SilentAgent(), [], emit)) == ""
    assert emitted == []


def test_stream_turn_passes_messages_payload():
    seen = []

    class CaptureAgent:
        async def astream_events(self, payload):
            seen.append(payload)
            yield {"event": "on_chat_model_end", "data": {"output": FakeMsg("ok")}}

    async def emit(evt):
        pass

    msgs = [{"role": "user", "content": "hi"}]
    assert asyncio.run(stream_agent_turn(CaptureAgent(), msgs, emit)) == "ok"
    assert seen == [{"messages": msgs}]


# ---- stream_agent_turn: astream path ---------------------------------------

def test_stream_turn_astream_fallback_uses_last_state():
    class AstreamAgent:
        async def astream(self, _input):
            yield {"messages": [FakeMsg("first")]}
            yield {"messages": [FakeMsg("first"), FakeMsg("second")]}

    async def emit(evt):
        raise AssertionError("astream path must not emit tokens")

    assert asyncio.run(stream_agent_turn(AstreamAgent(), [], emit)) == "second"


def test_stream_turn_astream_nested_state():
    class AstreamAgent:
        async def astream(self, _input):
            yield {"agent": {"messages": [FakeMsg("nested final")]}}
            yield {"messages": []}

    async def emit(evt):
        pass

    assert asyncio.run(stream_agent_turn(AstreamAgent(), [], emit)) == "nested final"


def test_stream_turn_astream_empty_states():
    class AstreamAgent:
        async def astream(self, _input):
            yield {}
            yield {"messages": []}

    async def emit(evt):
        pass

    assert asyncio.run(stream_agent_turn(AstreamAgent(), [], emit)) == ""


# ---- stream_agent_turn: ainvoke path ---------------------------------------

def test_stream_turn_ainvoke_dict_result():
    class AinvokeAgent:
        async def ainvoke(self, _input):
            return {"messages": [FakeMsg("reply")]}

    async def emit(evt):
        raise AssertionError("ainvoke path must not emit tokens")

    assert asyncio.run(stream_agent_turn(AinvokeAgent(), [], emit)) == "reply"


def test_stream_turn_ainvoke_empty_and_non_dict_results():
    class EmptyAgent:
        async def ainvoke(self, _input):
            return {"messages": []}

    class BareAgent:
        async def ainvoke(self, _input):
            return "not a dict"

    async def emit(evt):
        pass

    assert asyncio.run(stream_agent_turn(EmptyAgent(), [], emit)) == ""
    assert asyncio.run(stream_agent_turn(BareAgent(), [], emit)) == ""


def test_stream_turn_prefers_astream_events_over_astream():
    """An agent exposing both must take the token path, not astream."""
    emitted = []

    async def emit(evt):
        emitted.append(evt)

    class BothAgent:
        async def astream_events(self, _input):
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg("tok")}}

        async def astream(self, _input):
            yield {"messages": [FakeMsg("wrong path")]}

    assert asyncio.run(stream_agent_turn(BothAgent(), [], emit)) == "tok"
    assert emitted == [{"type": "token", "delta": "tok"}]
