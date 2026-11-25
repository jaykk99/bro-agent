"""Failover-robustness gaps, 2026-11-25 (continuing Nov 24 Horde hardening).

Each test reproduces a real bug on the old code and passes on the fix:

1. Horde poll loop: a 200 whose JSON is not an object (list/string) used
   to raise AttributeError out of _log_queue_progress(s).get() — skipping
   the job cancel (orphaned job on the worker) and aborting the host
   attempt outright. Non-dict JSON now counts as a bad poll like
   non-JSON does: tolerated once, cancels + fails over at the threshold.
2. Stale model list: a failed live-model refresh used to cache the
   hardcoded fallback for the full 6h, so a transient outage pinned
   every turn to stale models. Failures are now retried after 5 min.
3. Empty provider reply: the chat loop used to record an empty reply as
   a SUCCESS, never try the next provider, and tell Jay "all providers
   failed" when none had failed. Empty/whitespace-only replies now fail
   over like errors and refusals.
4. Train adapter files: a corrupt stored `bytes` value (non-numeric) in
   adapter_hashes_json used to raise ValueError out of int() and 500
   GET /api/train/{id} for the whole job; a non-string sha256 surfaced
   raw. Both now degrade per-entry.
5. Pollinations leg: ChatOpenAI was built with no request timeout, so a
   hung fallback leg held the turn for the full 600s turn timeout
   before failover could happen. The leg now has a 120s cap.
"""

import asyncio
import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-failover-2026-11-25-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("BRO_ACCESS_CODE", None)

from fastapi.testclient import TestClient  # noqa: E402

import app.agent as agent_mod  # noqa: E402
import app.main as main_mod  # noqa: E402
from app import horde_llm as horde  # noqa: E402
from app import rate_limit, train  # noqa: E402
from app.agent import _pollinations_model  # noqa: E402
from app.horde_llm import (  # noqa: E402
    _ahorde_generate_on,
    _horde_generate_on,
    _parse_status_payload,
)
from app.router import get_router  # noqa: E402

client = TestClient(main_mod.app)


@pytest.fixture(autouse=True)
def _reset_model_cache():
    horde._models_cache = None
    horde._models_refresh_failed_at = None
    yield
    horde._models_cache = None
    horde._models_refresh_failed_at = None


def _resp(payload, status=200):
    class R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        def json(self):
            return payload

    return R()


# ---- 1. non-dict JSON status payload (sync + async) ----


class _PollNonDictSync:
    """Poll always returns a 200 whose JSON is a list, not an object."""

    def __init__(self, *a, **k):
        self.deletes: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        return _resp({"id": "job-1"})

    def get(self, url, headers=None):
        return _resp(["not", "a", "dict"])

    def delete(self, url, headers=None):
        self.deletes.append(url)
        return _resp({})


class _PollNonDictAsync(_PollNonDictSync):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _resp({"id": "job-1"})

    async def get(self, url, headers=None):
        return _resp(["not", "a", "dict"])

    async def delete(self, url, headers=None):
        self.deletes.append(url)
        return _resp({})


def test_horde_poll_non_dict_json_cancels_at_threshold(monkeypatch):
    c = _PollNonDictSync()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: c)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="non-dict JSON"):
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    assert c.deletes == ["https://x.example/api/v2/generate/text/status/job-1"]


def test_ahorde_poll_non_dict_json_cancels_at_threshold(monkeypatch):
    c = _PollNonDictAsync()
    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: c)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    with pytest.raises(RuntimeError, match="non-dict JSON"):
        asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60))
    assert c.deletes == ["https://x.example/api/v2/generate/text/status/job-1"]


class _PollNonDictOnceSync(_PollNonDictSync):
    """One garbage poll, then the job finishes: transient garbage is tolerated."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._n = 0

    def get(self, url, headers=None):
        self._n += 1
        if self._n == 1:
            return _resp("just a string")
        return _resp({"done": True, "generations": [{"text": "recovered"}]})


class _PollNonDictOnceAsync(_PollNonDictOnceSync):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _resp({"id": "job-1"})

    async def get(self, url, headers=None):
        self._n += 1
        if self._n == 1:
            return _resp("just a string")
        return _resp({"done": True, "generations": [{"text": "recovered"}]})

    async def delete(self, url, headers=None):
        self.deletes.append(url)
        return _resp({})


def test_horde_poll_single_non_dict_json_is_tolerated(monkeypatch):
    c = _PollNonDictOnceSync()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: c)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    assert _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60) == "recovered"
    assert c.deletes == []  # job finished normally: nothing to cancel


def test_ahorde_poll_single_non_dict_json_is_tolerated(monkeypatch):
    c = _PollNonDictOnceAsync()
    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: c)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    assert asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)) == "recovered"
    assert c.deletes == []


def test_parse_status_payload_non_dict_is_runtime_error_not_attribute_error():
    with pytest.raises(RuntimeError, match="malformed status payload"):
        _parse_status_payload(["not", "a", "dict"])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="malformed status payload"):
        _parse_status_payload("just a string")  # type: ignore[arg-type]


# ---- 2. failed model refresh retries after 5 min, not 6h ----


def _fake_http_get(payload):
    def get(url, headers=None, timeout=None):
        return _resp(payload)

    return get


def _bad_http_get(*a, **k):
    raise RuntimeError("network down")


def test_resolve_models_failed_refresh_does_not_cache_fallback():
    got = horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=_bad_http_get)
    assert got == horde.HORDE_UNCENSORED_MODELS
    # The fallback must NOT be cached for 6h: the next call re-asks.
    assert horde._models_cache is None
    assert horde._models_refresh_failed_at is not None


def test_resolve_models_failed_refresh_retries_after_window():
    horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=_bad_http_get)
    calls: list[str] = []

    def good(url, headers=None, timeout=None):
        calls.append(url)
        return _resp([{"name": "koboldcpp/L3-8B-Stheno-v3.2", "count": 12}])

    # Inside the 5-minute retry window: no re-ask, fallback served.
    assert horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=good) == horde.HORDE_UNCENSORED_MODELS
    assert calls == []
    # Past the window: the live API is asked again and wins.
    horde._models_refresh_failed_at = time.time() - horde.MODELS_CACHE_RETRY_S - 1
    assert horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=good) == [
        "koboldcpp/L3-8B-Stheno-v3.2"
    ]
    assert calls != []


def test_invalidate_models_cache_clears_failed_refresh():
    horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=_bad_http_get)
    assert horde._models_refresh_failed_at is not None
    horde.invalidate_models_cache()
    assert horde._models_refresh_failed_at is None
    # A timeout-invalidate must re-ask immediately, not wait out the window.
    calls: list[str] = []

    def good(url, headers=None, timeout=None):
        calls.append(url)
        return _resp([{"name": "koboldcpp/L3-8B-Stheno-v3.2", "count": 12}])

    assert horde.resolve_models(horde.HORDE_UNCENSORED_MODELS, http_get=good) == [
        "koboldcpp/L3-8B-Stheno-v3.2"
    ]
    assert calls != []


# ---- 3. empty provider reply fails over ----


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeAgent:
    """Scripted agent: plan is a list of ('reply', text) or ('raise', exc)."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0

    async def ainvoke(self, _input):
        self.calls += 1
        kind, val = self.plan.pop(0)
        if kind == "raise":
            raise val
        return {"messages": [_FakeMsg(val)]}


def _install(monkeypatch, script):
    used = []

    async def fake_create(preferred=None):
        used.append(preferred)
        return _FakeAgent(script[preferred]), preferred, False

    monkeypatch.setattr(agent_mod, "get_cached_agent", fake_create)
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


def test_chat_empty_reply_fails_over_to_next_provider(monkeypatch):
    _fresh_router()
    used = _install(
        monkeypatch,
        {"pollinations": [("reply", "")], "horde": [("reply", "real answer")]},
    )
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    assert r.status_code == 200
    evts = _events(r)
    done = next(e for e in evts if e["type"] == "done")
    assert done["reply"] == "real answer" and done["model"] == "horde"
    assert used == ["pollinations", "horde"]
    assert any(e.get("type") == "progress" and e.get("stage") == "empty" for e in evts)
    assert get_router().status()["pollinations"]["consecutive_failures"] == 1


def test_chat_whitespace_only_reply_fails_over(monkeypatch):
    _fresh_router()
    used = _install(
        monkeypatch,
        {"pollinations": [("reply", "   \n  ")], "horde": [("reply", "real answer")]},
    )
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    done = next(e for e in _events(r) if e["type"] == "done")
    assert done["reply"] == "real answer"
    assert used == ["pollinations", "horde"]


def test_chat_empty_reply_from_all_providers_is_honest_error(monkeypatch):
    _fresh_router()
    used = _install(
        monkeypatch,
        {"pollinations": [("reply", "")], "horde": [("reply", "")]},
    )
    r = client.post("/api/chat", json={"message": "hi", "access_code": ""})
    evts = _events(r)
    err = next(e for e in evts if e["type"] == "error")
    assert "empty reply" in err["error"]
    assert "all providers failed" not in err["error"]
    assert used == ["pollinations", "horde"]


# ---- 4. corrupt adapter_files entries degrade per-entry ----


def _set_col(jid: str, col: str, value: str) -> None:
    c = train._connect()
    with train._lock:
        c.execute(f"UPDATE train_jobs SET {col}=? WHERE id=?", (value, jid))
        c.commit()


def test_row_to_dict_corrupt_adapter_bytes_degrades_to_zero():
    jid = train.create_job("model-corrupt-bytes")
    _set_col(
        jid,
        "adapter_hashes_json",
        json.dumps({"a.safetensors": {"sha256": "ab" * 32, "bytes": "not-a-number"}}),
    )
    job = train.get_job(jid)
    assert job["adapter_files"] == [{"name": "a.safetensors", "sha256": "ab" * 32, "bytes": 0}]


def test_row_to_dict_non_string_adapter_sha_skips_entry():
    jid = train.create_job("model-bad-sha")
    _set_col(
        jid,
        "adapter_hashes_json",
        json.dumps(
            {
                "bad.safetensors": {"sha256": 12345, "bytes": 10},
                "good.safetensors": {"sha256": "cd" * 32, "bytes": 20},
            }
        ),
    )
    job = train.get_job(jid)
    assert job["adapter_files"] == [{"name": "good.safetensors", "sha256": "cd" * 32, "bytes": 20}]


def test_row_to_dict_negative_adapter_bytes_clamps_to_zero():
    jid = train.create_job("model-negative-bytes")
    _set_col(
        jid,
        "adapter_hashes_json",
        json.dumps({"a.safetensors": {"sha256": "ab" * 32, "bytes": -50}}),
    )
    job = train.get_job(jid)
    assert job["adapter_files"] == [{"name": "a.safetensors", "sha256": "ab" * 32, "bytes": 0}]


# ---- 5. Pollinations leg has a per-request timeout ----


def test_pollinations_model_has_request_timeout():
    m = _pollinations_model()
    assert m.request_timeout == 120
    assert m.root_client.timeout == 120.0


def test_pollinations_timeout_env_override(monkeypatch):
    monkeypatch.setenv("BRO_POLLINATIONS_TIMEOUT_S", "30")
    assert _pollinations_model().request_timeout == 30


def test_pollinations_timeout_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("BRO_POLLINATIONS_TIMEOUT_S", "not-a-number")
    assert _pollinations_model().request_timeout == 120
