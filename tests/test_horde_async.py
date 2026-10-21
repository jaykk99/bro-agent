"""Tests for the Horde LLM module: tool-call parsing, prompt building, and the
new non-blocking async generate path.

langchain_core is NOT installed in this environment, so minimal stubs are
injected into sys.modules before importing app.horde_llm. The stubs mirror
the small slice of the langchain API the module actually uses.
"""

import asyncio
import os
import sys
import types
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _install_lc_stubs():
    from pydantic import BaseModel

    lc = types.ModuleType("langchain_core")
    callbacks = types.ModuleType("langchain_core.callbacks")
    lm = types.ModuleType("langchain_core.language_models")
    chat_models = types.ModuleType("langchain_core.language_models.chat_models")
    messages = types.ModuleType("langchain_core.messages")
    outputs = types.ModuleType("langchain_core.outputs")

    class CallbackManagerForLLMRun:  # noqa: D101
        pass

    @dataclass
    class BaseMessage:  # noqa: D101
        content: str = ""
        type: str = "base"

    @dataclass
    class HumanMessage(BaseMessage):
        type: str = "human"

    @dataclass
    class SystemMessage(BaseMessage):
        type: str = "system"

    @dataclass
    class ToolMessage(BaseMessage):
        type: str = "tool"
        name: str = ""

    @dataclass
    class AIMessage(BaseMessage):
        type: str = "ai"
        tool_calls: list = field(default_factory=list)

    @dataclass
    class ChatGeneration:  # noqa: D101
        message: AIMessage

    @dataclass
    class ChatResult:  # noqa: D101
        generations: list

    class BaseChatModel(BaseModel):
        """Pydantic-based stand-in: model_copy + PrivateAttr behave for real."""

        model_config = {"arbitrary_types_allowed": True}

        @property
        def _llm_type(self) -> str:
            return "stub"

    callbacks.CallbackManagerForLLMRun = CallbackManagerForLLMRun
    chat_models.BaseChatModel = BaseChatModel
    messages.AIMessage = AIMessage
    messages.BaseMessage = BaseMessage
    messages.HumanMessage = HumanMessage
    messages.SystemMessage = SystemMessage
    messages.ToolMessage = ToolMessage
    outputs.ChatGeneration = ChatGeneration
    outputs.ChatResult = ChatResult
    lm.chat_models = chat_models

    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.callbacks"] = callbacks
    sys.modules["langchain_core.language_models"] = lm
    sys.modules["langchain_core.language_models.chat_models"] = chat_models
    sys.modules["langchain_core.messages"] = messages
    sys.modules["langchain_core.outputs"] = outputs
    return messages


_msgs = _install_lc_stubs()

from app import horde_llm as horde  # noqa: E402
from app.horde_llm import (  # noqa: E402
    HordeChatModel,
    _ahorde_generate,
    _ahorde_generate_on,
    _messages_to_prompt,
    _parse_tool_calls,
    _tool_schema_text,
)


def test_parse_fenced_tool_call():
    text = 'Let me check.\n```tool_call\n{"name": "web_fetch", "arguments": {"url": "http://x.example/"}}\n```\nDone soon.'
    clean, calls = _parse_tool_calls(text)
    assert calls == [{"name": "web_fetch", "args": {"url": "http://x.example/"}}]
    assert "tool_call" not in clean
    assert "Done soon." in clean


def test_parse_multiple_fences():
    text = '```tool_call\n{"name": "a", "arguments": {}}\n```\n```tool_call\n{"name": "b", "arguments": {"x": 1}}\n```'
    _, calls = _parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_parse_bad_json_skipped():
    clean, calls = _parse_tool_calls("```tool_call\nnot json\n```\nhello")
    assert calls == []
    assert "hello" in clean


def test_parse_plain_text_no_calls():
    clean, calls = _parse_tool_calls("Just a normal reply.")
    assert calls == [] and clean == "Just a normal reply."


def test_messages_to_prompt_replays_tool_history():
    msgs = [
        _msgs.SystemMessage(content="You are bro."),
        _msgs.HumanMessage(content="fetch it"),
        _msgs.AIMessage(content="on it", tool_calls=[{"name": "web_fetch", "args": {"url": "http://x.example/"}}]),
        _msgs.ToolMessage(content="page text", name="web_fetch"),
    ]
    prompt = _messages_to_prompt(msgs, "Available tools:\n- web_fetch: fetch")
    assert "You are bro." in prompt
    assert "User: fetch it" in prompt
    assert "Tool result (web_fetch): page text" in prompt
    assert "tool_call" in prompt  # prior call replayed as text


def test_tool_schema_text():
    class FakeTool:
        name = "demo"
        description = "does things"

        class args_schema:  # minimal stand-in
            @staticmethod
            def model_json_schema():
                return {"properties": {"q": {"type": "string", "description": "query"}}, "required": ["q"]}

    text = _tool_schema_text([FakeTool()])
    assert "- demo: does things" in text
    assert "q*: string - query" in text


def test_generate_parses_tool_calls(monkeypatch):
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(
        horde,
        "_horde_generate",
        lambda *a, **k: 'Checking now.\n```tool_call\n{"name": "web_search", "arguments": {"query": "test"}}\n```',
    )
    model = HordeChatModel()
    result = model._generate([_msgs.HumanMessage(content="hi")])
    msg = result.generations[0].message
    assert "Checking now." in msg.content
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc["name"] == "web_search" and tc["args"] == {"query": "test"}
    assert tc["id"].startswith("call_")


def test_bind_tools_injects_schemas():
    class FakeTool:
        name = "t1"
        description = "tool one"
        args_schema = None

    model = HordeChatModel().bind_tools([FakeTool()])
    prompt = model._build_prompt([_msgs.HumanMessage(content="go")])
    assert "- t1: tool one" in prompt
    assert "```tool_call" in prompt  # format instructions present


# ---- async path ----


class _FakeResp:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    """Fake httpx.AsyncClient: submit ok, one pending poll, then done."""

    def __init__(self, *a, **k):
        self.polls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        assert url.endswith("/generate/text/async")
        return _FakeResp(202, {"id": "job-1"})

    async def get(self, url, headers=None):
        self.polls += 1
        if self.polls < 2:
            return _FakeResp(200, {"done": False, "wait_time": 1})
        return _FakeResp(200, {"done": True, "generations": [{"text": "async hello"}]})


async def _no_sleep(_s):
    return None


def test_ahorde_generate_on_polls_to_done(monkeypatch):
    monkeypatch.setattr(horde.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    text = asyncio.run(
        _ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    )
    assert text == "async hello"


def test_ahorde_generate_on_submit_403(monkeypatch):
    class BadClient(_FakeAsyncClient):
        async def post(self, url, headers=None, json=None):
            return _FakeResp(403, {})

    monkeypatch.setattr(horde.httpx, "AsyncClient", BadClient)
    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    try:
        asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {}, 60))
    except RuntimeError as e:
        assert "403" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_ahorde_generate_fails_over_to_next_host(monkeypatch):
    calls = []

    class FailThenOk(_FakeAsyncClient):
        async def post(self, url, headers=None, json=None):
            calls.append(url)
            if "stablehorde" in url:
                return _FakeResp(403, {})
            return await super().post(url, headers=headers, json=json)

    monkeypatch.setattr(horde, "HORDE_APIS", ["https://stablehorde.net/api/v2", "https://aihorde.net/api/v2"])
    monkeypatch.setattr(horde.httpx, "AsyncClient", FailThenOk)
    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    text = asyncio.run(_ahorde_generate("hi", ["m"], 8192, 500, 0.3, 60))
    assert text == "async hello"
    assert any("stablehorde" in u for u in calls)


def test_agenerate_end_to_end(monkeypatch):
    async def fake_gen(*a, **k):
        return 'Sure.\n```tool_call\n{"name": "file_read", "arguments": {"path": "a.txt"}}\n```'

    monkeypatch.setattr(horde, "_ahorde_generate", fake_gen)
    model = HordeChatModel()
    result = asyncio.run(model._agenerate([_msgs.HumanMessage(content="read it")]))
    msg = result.generations[0].message
    assert msg.content.strip() == "Sure."
    assert msg.tool_calls[0]["name"] == "file_read"


# ---- live model refresh ----

def _fake_http_get(payload, status=200):
    class R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        def json(self):
            return payload

    def get(url, headers=None, timeout=None):
        return R()

    return get


def test_resolve_models_custom_list_passthrough():
    horde._models_cache = None
    custom = ["koboldcpp/MyModel"]
    assert horde.resolve_models(custom) == custom


def test_resolve_models_live_refresh(monkeypatch):
    horde._models_cache = None
    payload = [
        {"name": "koboldcpp/L3-8B-Stheno-v3.2", "count": 12},
        {"name": "koboldcpp/dead-model", "count": 0},  # no workers -> excluded
        {"name": "koboldcpp/gpt4-aligned", "count": 30},  # not uncensored -> excluded
        {"name": "aphrodite/My-Heretic-Merge", "count": 5},
    ]
    got = horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=_fake_http_get(payload))
    assert got == ["koboldcpp/L3-8B-Stheno-v3.2", "aphrodite/My-Heretic-Merge"]
    # cached: second call must not hit the network (boom getter)
    def boom(*a, **k):
        raise AssertionError("should use cache")

    assert horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=boom) == got


def test_resolve_models_falls_back_on_api_failure():
    horde._models_cache = None

    def bad(*a, **k):
        raise RuntimeError("network down")

    got = horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=bad)
    assert got == horde.HORDE_UNCENSORED_MODELS
