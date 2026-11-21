"""Tests for the Horde failover hardening (2026-11-21):

  * adaptive poll interval from queue_position (fast when running, slow
    when deep in the queue),
  * one bounded Retry-After retry on a 429 submit,
  * stale live-model cache invalidation when a job times out,
  * LangChain `stop` sequences passed through as Horde stop_sequence.

Follows the fake-client style of tests/test_horde_async.py. Needs the
real langchain_core, like tests/test_horde.py.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage  # noqa: E402

from app import horde_llm as horde  # noqa: E402
from app.horde_llm import (  # noqa: E402
    HordeChatModel,
    _ahorde_generate,
    _ahorde_generate_on,
    _check_submit_response,
    _horde_generate,
    _horde_generate_on,
    _poll_interval_s,
    _retry_after_s,
    _submit_body,
)


class _Resp:
    """Minimal httpx-like response with a headers mapping."""

    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# ---- adaptive poll interval ----


def test_poll_interval_defaults():
    assert _poll_interval_s(None) == 4.0
    assert _poll_interval_s({}) == 4.0
    assert _poll_interval_s({"queue_position": None}) == 4.0
    assert _poll_interval_s({"queue_position": "junk"}) == 4.0


def test_poll_interval_running_is_fast():
    assert _poll_interval_s({"queue_position": 0}) == 2.0


def test_poll_interval_queued_is_slow():
    assert _poll_interval_s({"queue_position": 1}) == 6.0
    assert _poll_interval_s({"queue_position": 42}) == 6.0
    assert _poll_interval_s({"queue_position": "3"}) == 6.0


# ---- Retry-After parsing ----


def test_retry_after_parses_seconds():
    assert _retry_after_s(_Resp(429, headers={"Retry-After": "2"})) == 2.0
    assert _retry_after_s(_Resp(429, headers={"retry-after": "0.5"})) == 0.5


def test_retry_after_caps_huge_values():
    # An edge telling us to go away for 5 minutes must not park the turn:
    # cap at 10s, the router fails over after that.
    assert _retry_after_s(_Resp(429, headers={"Retry-After": "300"})) == 10.0


def test_retry_after_missing_or_bad_is_none():
    assert _retry_after_s(_Resp(429)) is None
    assert _retry_after_s(_Resp(429, headers={"Retry-After": "junk"})) is None
    assert _retry_after_s(_Resp(429, headers={"Retry-After": "0"})) is None
    assert _retry_after_s(_Resp(429, headers={"Retry-After": "-5"})) is None
    assert _retry_after_s(object()) is None  # no .headers at all


# ---- submit response check ----


def test_check_submit_response():
    _check_submit_response(_Resp(200))
    _check_submit_response(_Resp(202))
    for status, frag in ((403, "403"), (429, "429")):
        try:
            _check_submit_response(_Resp(status))
        except RuntimeError as e:
            assert frag in str(e)
        else:
            raise AssertionError(f"expected RuntimeError for {status}")
    try:
        _check_submit_response(_Resp(500))
    except RuntimeError as e:
        assert "500" in str(e)
    else:
        raise AssertionError("expected RuntimeError for 500")


# ---- stop passthrough ----


def test_submit_body_stop_sequence():
    body = _submit_body("hi", ["m"], 8192, 500, 0.3, stop=["END", "User:"])
    assert body["params"]["stop_sequence"] == ["END", "User:"]
    assert "stop_sequence" not in _submit_body("hi", ["m"], 8192, 500, 0.3)
    assert "stop_sequence" not in _submit_body("hi", ["m"], 8192, 500, 0.3, stop=[])


def test_generate_passes_stop_to_submit(monkeypatch):
    """_horde_generate puts the stop list into the posted JSON body."""
    posted: list[dict] = []

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            posted.append(json)
            return _Resp(202, {"id": "job-1"})

        def get(self, url, headers=None):
            return _Resp(200, {"done": True, "generations": [{"text": "ok"}]})

    monkeypatch.setattr(horde.httpx, "Client", Client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    text = _horde_generate("hi", ["m"], 8192, 500, 0.3, 60, stop=["END"])
    assert text == "ok"
    assert posted[0]["params"]["stop_sequence"] == ["END"]


def test_model_generate_passes_stop(monkeypatch):
    seen: dict = {}

    def fake_gen(*a, **k):
        seen.update(k)
        return "done"

    # Seed the live-model cache so resolve_models() never touches the
    # network when this file runs standalone.
    horde._models_cache = (time.time(), ["m"])
    monkeypatch.setattr(horde, "_horde_generate", fake_gen)
    model = HordeChatModel()
    model._generate([HumanMessage(content="hi")], stop=["END"])
    assert seen.get("stop") == ["END"]
    horde._models_cache = None


# ---- 429 Retry-After retry on submit ----


class _RateLimitedSyncClient:
    """First submit 429s with Retry-After, second submit succeeds."""

    def __init__(self, *a, retry_after="1", second_status=202, **k):
        self.posts = 0
        self._retry_after = retry_after
        self._second_status = second_status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        self.posts += 1
        if self.posts == 1:
            return _Resp(429, {}, headers={"Retry-After": self._retry_after})
        return _Resp(self._second_status, {"id": "job-1"})

    def get(self, url, headers=None):
        return _Resp(200, {"done": True, "generations": [{"text": "after retry"}]})

    def delete(self, url, headers=None):
        return _Resp(200, {})


def test_horde_submit_retries_once_after_retry_after(monkeypatch):
    sleeps: list[float] = []
    client = _RateLimitedSyncClient(retry_after="1")
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda s: sleeps.append(s))
    text = _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    assert text == "after retry"
    assert client.posts == 2
    assert sleeps[0] == 1.0  # the Retry-After wait, before the poll sleep


def test_horde_submit_retry_after_is_capped(monkeypatch):
    sleeps: list[float] = []
    client = _RateLimitedSyncClient(retry_after="300")
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda s: sleeps.append(s))
    text = _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    assert text == "after retry"
    assert sleeps[0] == 10.0


def test_horde_submit_429_without_retry_after_fails_over(monkeypatch):
    # No header -> no guess: raise at once so the caller tries the next host.
    class NoHeader(_RateLimitedSyncClient):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._retry_after = None

        def post(self, url, headers=None, json=None):
            self.posts += 1
            return _Resp(429, {})

    client = NoHeader()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    try:
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    except RuntimeError as e:
        assert "429" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert client.posts == 1


def test_horde_submit_second_429_raises(monkeypatch):
    client = _RateLimitedSyncClient(retry_after="1", second_status=429)
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    try:
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    except RuntimeError as e:
        assert "429" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert client.posts == 2  # exactly one retry, then give up to failover


class _RateLimitedAsyncClient:
    """Async twin of _RateLimitedSyncClient."""

    def __init__(self, *a, retry_after="1", **k):
        self.posts = 0
        self._retry_after = retry_after

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        self.posts += 1
        if self.posts == 1:
            return _Resp(429, {}, headers={"Retry-After": self._retry_after})
        return _Resp(202, {"id": "job-1"})

    async def get(self, url, headers=None):
        return _Resp(200, {"done": True, "generations": [{"text": "async after retry"}]})

    async def delete(self, url, headers=None):
        return _Resp(200, {})


async def _no_sleep(s):
    return None


def test_ahorde_submit_retries_once_after_retry_after(monkeypatch):
    client = _RateLimitedAsyncClient(retry_after="2")
    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: client)
    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    text = asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60))
    assert text == "async after retry"
    assert client.posts == 2


# ---- adaptive poll interval in the real loops ----


class _QueuedThenDoneSyncClient:
    """Polls: deep in queue (position 3), then done."""

    def __init__(self, *a, **k):
        self.polls = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        return _Resp(202, {"id": "job-1"})

    def get(self, url, headers=None):
        self.polls += 1
        if self.polls == 1:
            return _Resp(200, {"done": False, "queue_position": 3, "wait_time": 20})
        return _Resp(200, {"done": True, "generations": [{"text": "queued then done"}]})

    def delete(self, url, headers=None):
        return _Resp(200, {})


def test_horde_poll_interval_adapts_to_queue(monkeypatch):
    sleeps: list[float] = []
    client = _QueuedThenDoneSyncClient()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda s: sleeps.append(s))
    text = _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    assert text == "queued then done"
    # First poll: no status yet -> default 4s. Second poll: queue_position
    # 3 -> queued 6s. The old code slept a flat 4s both times.
    assert sleeps == [4.0, 6.0]


class _QueuedThenDoneAsyncClient:
    def __init__(self, *a, **k):
        self.polls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _Resp(202, {"id": "job-1"})

    async def get(self, url, headers=None):
        self.polls += 1
        if self.polls == 1:
            return _Resp(200, {"done": False, "queue_position": 0, "wait_time": 2})
        return _Resp(200, {"done": True, "generations": [{"text": "running then done"}]})

    async def delete(self, url, headers=None):
        return _Resp(200, {})


def test_ahorde_poll_interval_fast_when_running(monkeypatch):
    sleeps: list[float] = []
    client = _QueuedThenDoneAsyncClient()

    async def record(s):
        sleeps.append(s)

    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: client)
    monkeypatch.setattr(horde.asyncio, "sleep", record)
    text = asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60))
    assert text == "running then done"
    # First poll default 4s; job then reports queue_position 0 (running)
    # so the next poll comes after 2s, not 4s.
    assert sleeps == [4.0, 2.0]


# ---- stale model cache invalidation on timeout ----


class _PendingSyncClient:
    """Submit ok, polls stay pending forever."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        return _Resp(202, {"id": "job-1"})

    def get(self, url, headers=None):
        return _Resp(200, {"done": False, "queue_position": 3, "wait_time": 42})

    def delete(self, url, headers=None):
        return _Resp(200, {})


class _PendingAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _Resp(202, {"id": "job-1"})

    async def get(self, url, headers=None):
        return _Resp(200, {"done": False, "queue_position": 3, "wait_time": 42})

    async def delete(self, url, headers=None):
        return _Resp(200, {})


def _seed_cache():
    horde._models_cache = (time.time(), ["koboldcpp/cached-model"])


def test_horde_timeout_invalidates_model_cache(monkeypatch):
    _seed_cache()
    client = _PendingSyncClient()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    try:
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 0)
    except RuntimeError as e:
        assert "timed out" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert horde._models_cache is None


def test_ahorde_timeout_invalidates_model_cache(monkeypatch):
    _seed_cache()
    client = _PendingAsyncClient()
    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: client)
    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    try:
        asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 0))
    except RuntimeError as e:
        assert "timed out" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert horde._models_cache is None


def test_success_keeps_model_cache(monkeypatch):
    # Only a timeout invalidates: a healthy turn must not force a
    # live refresh on the next call.
    _seed_cache()
    client = _QueuedThenDoneSyncClient()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    text = _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    assert text == "queued then done"
    assert horde._models_cache is not None
    assert horde._models_cache[1] == ["koboldcpp/cached-model"]
    horde._models_cache = None  # leave no state for sibling tests


def test_poll_fail_fast_keeps_model_cache(monkeypatch):
    # A sick status endpoint is a host problem, not dead models: the
    # cache stays, and host failover handles it.
    _seed_cache()

    class Sick(_PendingSyncClient):
        def get(self, url, headers=None):
            return _Resp(500)

    client = Sick()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    try:
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    except RuntimeError as e:
        assert "status poll failed" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert horde._models_cache is not None
    horde._models_cache = None


def test_invalidate_models_cache_is_idempotent():
    horde._models_cache = None
    horde.invalidate_models_cache()
    assert horde._models_cache is None
    horde._models_cache = (time.time(), ["x"])
    horde.invalidate_models_cache()
    assert horde._models_cache is None
