"""Pin the remaining untested paths in app/horde_llm.py.

Coverage gaps left after test_horde.py / test_horde_async.py /
test_horde_retry.py: live-model refresh with no uncensored models,
_tool_schema_text's exception branch, the generic-message branch of
_messages_to_parts, _fit_prompt's header-overflow path, the lenient
_bare-JSON fallback skips in _parse_tool_calls, _parse_status_payload
fault/empty paths, the no-job-id submit paths (sync + async), the async
non-JSON poll fail-fast, and the invalid HORDE_TIMEOUT_S fallback.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import horde_llm as horde  # noqa: E402
from app.horde_llm import (  # noqa: E402
    HORDE_CHAT_TIMEOUT_S,
    HordeChatModel,
    _ahorde_generate,
    _ahorde_generate_on,
    _default_horde_timeout,
    _fetch_live_models,
    _fit_prompt,
    _horde_generate_on,
    _messages_to_parts,
    _parse_status_payload,
    _parse_tool_calls,
    _tool_schema_text,
)


# ---- _fetch_live_models: no uncensored models anywhere ----


def _resp(payload, status=200):
    class R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        def json(self):
            return payload

    return R()


def test_fetch_live_models_no_uncensored_anywhere_returns_empty():
    # Both hosts answer, but neither lists an online uncensored model:
    # the refresh returns [] (the caller treats it as "no refresh",
    # same as before) instead of raising or returning junk.
    def get(url, headers=None, timeout=None):
        if "stablehorde" in url:
            return _resp([{"name": "meta-llama/Llama-3-8B-Instruct", "count": 9}])
        return _resp([{"name": "mistralai/Mistral-7B", "count": 0}])

    assert _fetch_live_models(http_get=get) == []


def test_fetch_live_models_non_list_payload_returns_empty():
    # A dict (or any non-list) payload has no model items at all.
    def get(url, headers=None, timeout=None):
        return _resp({"error": "not a list"})

    assert _fetch_live_models(http_get=get) == []


def test_fetch_live_models_zero_workers_skipped():
    # A matching name with zero workers is not live.
    def get(url, headers=None, timeout=None):
        return _resp([{"name": "koboldcpp/L3-8B-Stheno-v3.2", "count": 0}])

    assert _fetch_live_models(http_get=get) == []


# ---- _tool_schema_text: schema introspection failure ----


def test_tool_schema_text_schema_error_degrades_to_no_args():
    class ExplodingSchema:
        def model_json_schema(self):
            raise RuntimeError("schema boom")

    class FakeTool:
        name = "boom"
        description = "goes bang"
        args_schema = ExplodingSchema()

    text = _tool_schema_text([FakeTool()])
    assert text == "- boom: goes bang"  # no args line, no raise


def test_tool_schema_text_dict_args_without_schema():
    class FakeTool:
        name = "d"
        description = "dict args"
        args_schema = None
        args = {"path": {"type": "string", "description": "file path"}}

    assert _tool_schema_text([FakeTool()]) == "- d: dict args\n  args: path: string - file path"


# ---- _messages_to_parts: generic message type ----


def test_messages_to_parts_generic_message_type():
    try:
        from langchain_core.messages import ChatMessage
    except ImportError:
        import pytest

        pytest.skip("langchain_core ChatMessage unavailable")
    header, convo = _messages_to_parts(
        [ChatMessage(content="hey", role="user")], tools_text=""
    )
    assert convo == ["chat: hey"]
    assert header == ""


# ---- _fit_prompt: header alone overflows ----


def test_fit_prompt_header_overflow_keeps_newest_tail():
    # Header + tail already exceed the budget: keep the newest content.
    result = _fit_prompt("H" * 100, ["x"], 20)
    assert len(result) == 20
    assert result.endswith("\nAssistant:")
    # The newest chars of the prompt survive; the old header is cut.
    assert result.startswith("ssistant:")


def test_fit_prompt_exact_fit_unchanged():
    header, convo = "H" * 10, ["a", "b"]
    full = header + "\n".join(convo) + "\nAssistant:"
    assert _fit_prompt(header, convo, len(full)) == full


# ---- _parse_tool_calls: lenient fallback skips ----


def test_parse_bare_json_skips_invalid_json_then_parses():
    # A "{" that fails raw_decode is skipped, not fatal.
    text = '{not json {"name": "t", "arguments": {"a": 1}}'
    clean, calls = _parse_tool_calls(text)
    assert calls == [{"name": "t", "args": {"a": 1}}]
    assert clean == "{not json"


def test_parse_bare_json_skips_nameless_object():
    # An object with no "name" is skipped; the next bare object parses.
    text = '{"foo": 1} {"name": "t2", "arguments": {}}'
    clean, calls = _parse_tool_calls(text)
    assert calls == [{"name": "t2", "args": {}}]


def test_parse_bare_json_skips_non_string_name():
    text = '{"name": 42, "arguments": {}} {"name": "ok", "arguments": {}}'
    _, calls = _parse_tool_calls(text)
    assert calls == [{"name": "ok", "args": {}}]


# ---- _parse_status_payload ----


def test_parse_status_payload_faulted():
    try:
        _parse_status_payload({"faulted": True})
    except RuntimeError as e:
        assert "faulted" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_parse_status_payload_done_empty_generation():
    try:
        _parse_status_payload({"done": True, "generations": []})
    except RuntimeError as e:
        assert "empty generation" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_parse_status_payload_done_empty_text():
    try:
        _parse_status_payload({"done": True, "generations": [{"text": ""}]})
    except RuntimeError as e:
        assert "empty generation" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_parse_status_payload_pending_is_none():
    assert _parse_status_payload({"done": False}) is None
    assert _parse_status_payload({"done": True, "generations": [{"text": "hi"}]}) == "hi"


# ---- submit with no job id (sync + async) ----


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _NoIdSyncClient:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        return _Resp(202, {})  # 202 but no "id"

    def get(self, url, headers=None):
        raise AssertionError("must not poll without a job id")

    def delete(self, url, headers=None):
        return _Resp(200, {})


class _NoIdAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _Resp(202, {"id": ""})  # empty id counts as missing

    async def get(self, url, headers=None):
        raise AssertionError("must not poll without a job id")

    async def delete(self, url, headers=None):
        return _Resp(200, {})


def test_horde_generate_on_no_job_id(monkeypatch):
    monkeypatch.setattr(horde.httpx, "Client", _NoIdSyncClient)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    try:
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    except RuntimeError as e:
        assert "no job id" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_ahorde_generate_on_no_job_id(monkeypatch):
    monkeypatch.setattr(horde.httpx, "AsyncClient", _NoIdAsyncClient)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    try:
        asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60))
    except RuntimeError as e:
        assert "no job id" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


# ---- async non-JSON poll fail-fast (sync twin already pinned) ----


class _BadJsonResp:
    status_code = 200

    def json(self):
        raise ValueError("not JSON")

    def raise_for_status(self):
        return None


class _NonJsonAlwaysAsync:
    def __init__(self, *a, **k):
        self.deletes: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _Resp(202, {"id": "job-1"})

    async def get(self, url, headers=None):
        return _BadJsonResp()

    async def delete(self, url, headers=None):
        self.deletes.append(url)
        return _Resp(200, {})


def test_ahorde_generate_on_fails_fast_on_repeated_non_json_poll(monkeypatch):
    client = _NonJsonAlwaysAsync()
    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: client)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    try:
        asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60))
    except RuntimeError as e:
        assert "non-JSON" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert len(client.deletes) == 1  # job cancelled on fail-fast


# ---- async generate: all hosts down ----


class _AllDownAsync(_NoIdAsyncClient):
    async def post(self, url, headers=None, json=None):
        return _Resp(500, {})


def test_ahorde_generate_all_hosts_down_raises(monkeypatch):
    monkeypatch.setattr(
        horde, "HORDE_APIS", ["https://stablehorde.net/api/v2", "https://aihorde.net/api/v2"]
    )
    monkeypatch.setattr(horde.httpx, "AsyncClient", _AllDownAsync)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    try:
        asyncio.run(_ahorde_generate("hi", ["m"], 8192, 500, 0.3, 60))
    except RuntimeError as e:
        assert "all endpoints failed" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


# ---- _default_horde_timeout ----


def test_default_horde_timeout_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("HORDE_TIMEOUT_S", "not-a-number")
    assert _default_horde_timeout() == HORDE_CHAT_TIMEOUT_S


def test_default_horde_timeout_env_and_floor(monkeypatch):
    monkeypatch.setenv("HORDE_TIMEOUT_S", "120")
    assert _default_horde_timeout() == 120
    monkeypatch.setenv("HORDE_TIMEOUT_S", "0")
    assert _default_horde_timeout() == 1  # max(1, ...)
    monkeypatch.delenv("HORDE_TIMEOUT_S")
    assert _default_horde_timeout() == HORDE_CHAT_TIMEOUT_S


def test_horde_model_default_timeout_uses_env(monkeypatch):
    monkeypatch.setenv("HORDE_TIMEOUT_S", "90")
    assert HordeChatModel().timeout_s == 90
