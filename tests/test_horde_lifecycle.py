"""Pin Horde job-lifecycle edges in app/horde_llm.py.

Gaps the earlier hardening missed:

* A transport error (connect error, read timeout) from the status-poll
  GET propagated without cancelling the job — the job stayed
  queued/running on a community worker for an answer nobody reads.
  Both the sync and async poll loops now cancel first, then raise so
  the caller fails over to the next host.
* A Ctrl-C mid-poll in the sync path (CLI) orphaned the job the same
  way; it now cancels before the interrupt propagates.
* Status/submit payloads come from a community API: a non-dict submit
  body, or a done payload whose generations are a non-list, hold
  non-dicts, or carry a non-string "text", used to raise AttributeError
  (or hand a non-str downstream). They now read as "no job id" /
  "empty generation".
* One malformed entry in the live-model list aborted the whole
  refresh; malformed entries are now skipped.
"""

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import horde_llm as horde  # noqa: E402
from app.horde_llm import (  # noqa: E402
    _ahorde_generate,
    _ahorde_generate_on,
    _fetch_live_models,
    _horde_generate_on,
    _parse_status_payload,
)


def _resp(payload, status=200):
    class R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        def json(self):
            return payload

    return R()


def _connect_error():
    return httpx.ConnectError("boom", request=httpx.Request("GET", "https://x.example"))


# ---- _parse_status_payload: malformed generations ----


def test_parse_status_payload_generations_not_a_list():
    with pytest.raises(RuntimeError, match="empty generation"):
        _parse_status_payload({"done": True, "generations": {"text": "hi"}})


def test_parse_status_payload_generation_not_a_dict():
    with pytest.raises(RuntimeError, match="empty generation"):
        _parse_status_payload({"done": True, "generations": ["not a dict"]})


def test_parse_status_payload_generation_text_not_a_string():
    with pytest.raises(RuntimeError, match="empty generation"):
        _parse_status_payload({"done": True, "generations": [{"text": 123}]})


def test_parse_status_payload_valid_still_returns_text():
    assert _parse_status_payload({"done": True, "generations": [{"text": "hi"}]}) == "hi"


# ---- _fetch_live_models: malformed entries skipped ----


def test_fetch_live_models_skips_malformed_items():
    def get(url, headers=None, timeout=None):
        return _resp(
            [
                "not a dict",
                {"name": "koboldcpp/L3-8B-Stheno-v3.2", "count": "abc"},
                {"name": "koboldcpp/L3-8B-Stheno-v3.2", "count": "3"},
                {"name": "koboldcpp/Rocinante-X-12B-v1", "count": 2},
                {"count": 5},  # no name at all
            ]
        )

    # The two well-formed uncensored entries survive; the malformed ones
    # (and the nameless one, which matches no hint) do not abort the run.
    assert _fetch_live_models(http_get=get) == [
        "koboldcpp/L3-8B-Stheno-v3.2",
        "koboldcpp/Rocinante-X-12B-v1",
    ]


# ---- submit with a non-dict JSON body ----


class _NonDictSubmitSync:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        return _resp(["not", "a", "dict"])

    def get(self, url, headers=None):
        raise AssertionError("must not poll without a job id")

    def delete(self, url, headers=None):
        return _resp({})


class _NonDictSubmitAsync(_NonDictSubmitSync):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _resp("just a string")

    async def get(self, url, headers=None):
        raise AssertionError("must not poll without a job id")

    async def delete(self, url, headers=None):
        return _resp({})


def test_horde_generate_on_non_dict_submit_body_is_no_job_id(monkeypatch):
    monkeypatch.setattr(horde.httpx, "Client", _NonDictSubmitSync)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="no job id"):
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)


def test_ahorde_generate_on_non_dict_submit_body_is_no_job_id(monkeypatch):
    monkeypatch.setattr(horde.httpx, "AsyncClient", _NonDictSubmitAsync)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    with pytest.raises(RuntimeError, match="no job id"):
        asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60))


# ---- poll transport error cancels the job (sync + async) ----


class _PollDiesSync:
    def __init__(self, *a, **k):
        self.deletes: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        return _resp({"id": "job-9"})

    def get(self, url, headers=None):
        raise _connect_error()

    def delete(self, url, headers=None):
        self.deletes.append(url)
        return _resp({})


class _PollDiesAsync(_PollDiesSync):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _resp({"id": "job-9"})

    async def get(self, url, headers=None):
        raise _connect_error()

    async def delete(self, url, headers=None):
        self.deletes.append(url)
        return _resp({})


def test_horde_generate_on_poll_transport_error_cancels_job(monkeypatch):
    client = _PollDiesSync()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="transport error"):
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    assert client.deletes == ["https://x.example/api/v2/generate/text/status/job-9"]


def test_ahorde_generate_on_poll_transport_error_cancels_job(monkeypatch):
    client = _PollDiesAsync()
    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: client)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    with pytest.raises(RuntimeError, match="transport error"):
        asyncio.run(_ahorde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60))
    assert client.deletes == ["https://x.example/api/v2/generate/text/status/job-9"]


def test_ahorde_generate_fails_over_after_poll_transport_error(monkeypatch):
    """A dead status endpoint on host 1 cancels that job and the call
    succeeds on host 2 instead of dying with the transport error."""

    class _FailoverAsync:
        def __init__(self, *a, **k):
            self.deletes: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return _resp({"id": "job-1"})

        async def get(self, url, headers=None):
            if "stablehorde" in url:
                raise _connect_error()
            return _resp({"done": True, "generations": [{"text": "second host wins"}]})

        async def delete(self, url, headers=None):
            self.deletes.append(url)
            return _resp({})

    client = _FailoverAsync()
    monkeypatch.setattr(horde, "HORDE_APIS", ["https://stablehorde.net/api/v2", "https://aihorde.net/api/v2"])
    monkeypatch.setattr(horde.httpx, "AsyncClient", lambda *a, **k: client)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(horde.asyncio, "sleep", _no_sleep)
    result = asyncio.run(_ahorde_generate("hi", ["m"], 8192, 500, 0.3, 60))
    assert result == "second host wins"
    assert len(client.deletes) == 1  # the host-1 job was cancelled, not orphaned


# ---- Ctrl-C mid-poll cancels the job (sync path) ----


class _PollInterruptedSync(_PollDiesSync):
    def get(self, url, headers=None):
        raise KeyboardInterrupt


def test_horde_generate_on_keyboard_interrupt_cancels_job(monkeypatch):
    client = _PollInterruptedSync()
    monkeypatch.setattr(horde.httpx, "Client", lambda *a, **k: client)
    monkeypatch.setattr(horde.time, "sleep", lambda _s: None)
    with pytest.raises(KeyboardInterrupt):
        _horde_generate_on("https://x.example/api/v2", {"apikey": "k"}, {"prompt": "hi"}, 60)
    assert client.deletes == ["https://x.example/api/v2/generate/text/status/job-9"]
