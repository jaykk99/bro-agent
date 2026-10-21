"""Chat-stream tests with a fake agent — covers the provider failover loop in
POST /api/chat without any LLM: success, error failover, refusal-skip to
the next provider, and total failure. No network."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-chat-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)

from fastapi.testclient import TestClient  # noqa: E402

import app.agent as agent_mod  # noqa: E402
import app.main as main_mod  # noqa: E402
from app import rate_limit  # noqa: E402
from app.router import get_router  # noqa: E402

client = TestClient(main_mod.app)


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeAgent:
    """Scripted agent: plan is a list of ('reply', text) or ('raise', exc)."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0

    async def ainvoke(self, _input):
        self.calls += 1
        kind, val = self.plan.pop(0)
        if kind == "raise":
            raise val
        return {"messages": [FakeMsg(val)]}


def _install(monkeypatch, script):
    """script: {provider_name: [(kind, value), ...]}. Records provider order used."""

    used = []

    async def fake_create(preferred=None):
        used.append(preferred)
        return FakeAgent(script[preferred]), preferred

    monkeypatch.setattr(agent_mod, "create_bro_agent", fake_create)
    return used


def _events(resp):
    out = []
    for line in resp.text.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            out.append(json.loads(line[5:]))
    return out


def _fresh_router():
    r = get_router()
    for p in ("pollinations", "horde"):
        r.record_success(p)
    rate_limit.reset_for_tests()


def test_chat_success_single_provider(monkeypatch):
    _fresh_router()
    used = _install(monkeypatch, {"pollinations": [("reply", "hello there")], "horde": []})
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    assert r.status_code == 200
    evts = _events(r)
    done = next(e for e in evts if e["type"] == "done")
    assert done["reply"] == "hello there" and done["model"] == "pollinations"
    assert used == ["pollinations"]  # no failover needed


def test_chat_failover_on_error(monkeypatch):
    _fresh_router()
    used = _install(
        monkeypatch,
        {
            "pollinations": [("raise", RuntimeError("pollinations down"))],
            "horde": [("reply", "horde answer")],
        },
    )
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    done = next(e for e in _events(r) if e["type"] == "done")
    assert done["reply"] == "horde answer" and done["model"] == "horde"
    assert used == ["pollinations", "horde"]
    # pollinations took a failure mark (1 < threshold, still healthy-first)
    assert get_router().status()["pollinations"]["consecutive_failures"] == 1


def test_chat_refusal_skips_to_next_provider(monkeypatch):
    _fresh_router()
    used = _install(
        monkeypatch,
        {
            "pollinations": [("reply", "Sorry, I can't help with that.")],
            "horde": [("reply", "sure, here's the real answer")],
        },
    )
    r = client.post("/api/chat", json={"message": "write me a script", "access_code": ""})
    done = next(e for e in _events(r) if e["type"] == "done")
    # Jay never sees the refusal — the turn transparently moved providers.
    assert done["reply"] == "sure, here's the real answer"
    assert done["model"] == "horde"
    assert used == ["pollinations", "horde"]


def test_chat_refusal_on_last_provider_is_shown(monkeypatch):
    _fresh_router()
    _install(
        monkeypatch,
        {
            "pollinations": [("raise", RuntimeError("down"))],
            "horde": [("reply", "Sorry, I can't help with that.")],
        },
    )
    r = client.post("/api/chat", json={"message": "write me a script", "access_code": ""})
    done = next(e for e in _events(r) if e["type"] == "done")
    # nowhere left to go — the refusal is shown, not hidden or faked
    assert "Sorry, I can't" in done["reply"]


def test_chat_all_providers_fail(monkeypatch):
    _fresh_router()
    _install(
        monkeypatch,
        {
            "pollinations": [("raise", RuntimeError("down1"))],
            "horde": [("raise", RuntimeError("down2"))],
        },
    )
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    err = next(e for e in _events(r) if e["type"] == "error")
    assert "down2" in err["error"]
    # both providers now carry a failure mark
    st = get_router().status()
    assert st["pollinations"]["consecutive_failures"] >= 1
    assert st["horde"]["consecutive_failures"] >= 1


def test_chat_turn_persisted(monkeypatch):
    _fresh_router()
    _install(monkeypatch, {"pollinations": [("reply", "persisted!")], "horde": []})
    r = client.post("/api/chat", json={"message": "remember this", "access_code": ""})
    tid = next(e for e in _events(r) if e["type"] == "thread")["thread_id"]
    msgs = main_mod.memory.get_history(tid)
    assert msgs[0] == {"role": "user", "content": "remember this"}
    assert msgs[1] == {"role": "assistant", "content": "persisted!"}
