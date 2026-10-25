"""Chat-speed fixes: agent caching, real token streaming, Horde cap.

No network, no real LLM: fake agents stand in for UniHarness, and the
Horde hang test uses a fake httpx client that never reports done.
"""

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-chat-speed-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)

from fastapi.testclient import TestClient  # noqa: E402

import app.agent as agent_mod  # noqa: E402
import app.main as main_mod  # noqa: E402
from app import horde_llm as horde_mod  # noqa: E402
from app import rate_limit  # noqa: E402
from app.chat_stream import end_text, stream_agent_turn, stream_delta  # noqa: E402
from app.router import get_router  # noqa: E402

client = TestClient(main_mod.app)


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeStreamingAgent:
    """Yields token chunks + end events like LangGraph astream_events v2."""

    def __init__(self, chunks, final):
        self.chunks = list(chunks)
        self.final = final

    async def astream_events(self, _input):
        for c in self.chunks:
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg(c)}}
        yield {"event": "on_chat_model_end", "data": {"output": FakeMsg(self.final)}}


class FakeAinvokeAgent:
    def __init__(self, reply):
        self.reply = reply

    async def ainvoke(self, _input):
        return {"messages": [FakeMsg(self.reply)]}


def _fresh_router():
    r = get_router()
    for p in ("pollinations", "horde"):
        r.record_success(p)
    rate_limit.reset_for_tests()


def _events(resp):
    out = []
    for line in resp.text.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            out.append(json.loads(line[5:]))
    return out


# ---- 1. Agent caching ----

def test_cached_agent_reused_across_turns(monkeypatch):
    agent_mod.invalidate_agent_cache()
    builds = []

    async def fake_build(preferred=None):
        agent = FakeAinvokeAgent("x")
        builds.append(agent)
        return agent, preferred or "horde"

    monkeypatch.setattr(agent_mod, "create_bro_agent", fake_build)
    a1, s1, hit1 = asyncio.run(agent_mod.get_cached_agent(preferred="horde"))
    a2, s2, hit2 = asyncio.run(agent_mod.get_cached_agent(preferred="horde"))
    try:
        assert a1 is a2, "second turn must reuse the cached agent object"
        assert len(builds) == 1, f"agent built {len(builds)} times, expected 1"
        assert hit1 is False and hit2 is True
        assert s1 == s2 == "horde"
    finally:
        agent_mod.invalidate_agent_cache()


def test_cache_invalidation_rebuilds(monkeypatch):
    agent_mod.invalidate_agent_cache()
    builds = []

    async def fake_build(preferred=None):
        builds.append(1)
        return FakeAinvokeAgent("x"), "horde"

    monkeypatch.setattr(agent_mod, "create_bro_agent", fake_build)
    asyncio.run(agent_mod.get_cached_agent(preferred="horde"))
    assert agent_mod.invalidate_agent_cache() == 1
    asyncio.run(agent_mod.get_cached_agent(preferred="horde"))
    assert len(builds) == 2, "invalidate must force a rebuild on next turn"
    agent_mod.invalidate_agent_cache()


def test_cache_fingerprint_tracks_workdir(monkeypatch):
    # monkeypatch restores the env on teardown: other test modules
    # (test_ops) read BRO_WORKDIR at test time and must not see our pop.
    baseline = agent_mod.agent_cache_fingerprint("horde")
    monkeypatch.setenv("BRO_WORKDIR", "/tmp/bro-work-test-a")
    changed = agent_mod.agent_cache_fingerprint("horde")
    assert baseline != changed, "workdir change must not reuse a stale cached agent"
    assert agent_mod.agent_cache_fingerprint("horde") == changed


# ---- 2. Token streaming ----

def test_stream_deltas_arrive_and_final_is_last_end():
    emitted = []

    async def emit(evt):
        emitted.append(evt)

    # Intermediate model turn (planning narration) streams, then the final
    # turn streams — final text must be the LAST end, not the concat.
    class TwoTurnAgent:
        async def astream_events(self, _input):
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg("planning...")}}
            yield {"event": "on_chat_model_end", "data": {"output": FakeMsg("planning...")}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg("final")}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeMsg(" answer")}}
            yield {"event": "on_chat_model_end", "data": {"output": FakeMsg("final answer")}}

    final = asyncio.run(stream_agent_turn(TwoTurnAgent(), [], emit))
    assert [e["delta"] for e in emitted] == ["planning...", "final", " answer"]
    assert all(e["type"] == "token" for e in emitted)
    assert final == "final answer"


def test_stream_helpers_ignore_non_model_events():
    assert stream_delta({"event": "on_tool_start", "data": {}}) is None
    assert end_text({"event": "on_chat_model_stream", "data": {}}) is None
    assert end_text(
        {"event": "on_chat_model_end", "data": {"output": FakeMsg("done text")}}
    ) == "done text"


def test_http_token_events_precede_done_with_unchanged_done(monkeypatch):
    _fresh_router()

    async def fake_cached(preferred=None):
        return FakeStreamingAgent(["hel", "lo ", "there"], "hello there"), preferred, False

    monkeypatch.setattr(agent_mod, "get_cached_agent", fake_cached)
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    assert r.status_code == 200
    evts = _events(r)
    kinds = [e["type"] for e in evts]
    assert kinds.index("token") < kinds.index("done"), "tokens must arrive before done"
    assert "".join(e["delta"] for e in evts if e["type"] == "token") == "hello there"
    done = next(e for e in evts if e["type"] == "done")
    # done payload shape unchanged from before this change
    assert done["reply"] == "hello there" and done["model"] == "horde"
    assert set(done) == {"type", "reply", "model"}


def test_http_ainvoke_only_agent_has_no_fake_tokens(monkeypatch):
    _fresh_router()

    async def fake_cached(preferred=None):
        return FakeAinvokeAgent("plain reply"), preferred, False

    monkeypatch.setattr(agent_mod, "get_cached_agent", fake_cached)
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    evts = _events(r)
    assert not any(e["type"] == "token" for e in evts), "no synthetic tokens for ainvoke-only agents"
    assert next(e for e in evts if e["type"] == "done")["reply"] == "plain reply"


# ---- 3. Horde cap ----

def test_horde_default_timeout_capped_well_below_240():
    assert horde_mod.HORDE_CHAT_TIMEOUT_S == 60
    assert horde_mod.HordeChatModel().timeout_s == 60


def test_horde_timeout_env_override(monkeypatch):
    monkeypatch.setenv("HORDE_TIMEOUT_S", "45")
    assert horde_mod.HordeChatModel().timeout_s == 45


def test_hung_horde_call_aborts_at_cap(monkeypatch):
    """A Horde job that never reports done must raise by the cap, not 240s."""
    monkeypatch.setattr(horde_mod, "HORDE_APIS", ["https://horde-test.invalid/api/v2"])

    class Resp:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class NeverDoneClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return Resp({"id": "job-never-done"})

        async def get(self, *a, **k):
            return Resp({"done": False, "faulted": False})

    monkeypatch.setattr(horde_mod.httpx, "AsyncClient", NeverDoneClient)
    start = time.monotonic()
    with __import__("pytest").raises(RuntimeError, match="timed out"):
        asyncio.run(horde_mod._ahorde_generate("prompt", ["m"], 8192, 100, 0.3, 5))
    elapsed = time.monotonic() - start
    # 5s cap + 4s poll granularity: must abort well under the old 240s.
    assert elapsed < 40, f"hung Horde call took {elapsed:.1f}s, cap not enforced"
