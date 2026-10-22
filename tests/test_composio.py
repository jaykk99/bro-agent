"""Tests for the Composio tools. HTTP is faked; no network, no key needed."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.tools_composio as tc  # noqa: E402
from app.tools_composio import (  # noqa: E402
    ComposioAppsTool,
    ComposioExecuteTool,
    ComposioListToolsTool,
)


class FakeResp:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    @property
    def text(self):
        return str(self._payload)


class FakeClient:
    """Fake httpx.AsyncClient driven by a handler fn."""

    handler = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        return FakeClient.handler("POST", url, json)

    async def get(self, url, params=None, headers=None):
        return FakeClient.handler("GET", url, params)


def _install(handler, monkeypatch):
    FakeClient.handler = handler
    monkeypatch.setattr(tc.httpx, "AsyncClient", FakeClient)


def run(tool, monkeypatch, handler, key="k", **kwargs):
    monkeypatch.setenv("COMPOSIO_API_KEY", key)
    _install(handler, monkeypatch)
    # The real UniHarness BaseAgentTool has no .run(); tools are invoked
    # through __call__, which validates kwargs against args_schema.
    return asyncio.run(tool(**kwargs)).output


def test_execute_no_key(monkeypatch):
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    # The real UniHarness BaseAgentTool has no .run(); tools are invoked
    # through __call__, which validates kwargs against args_schema.
    out = asyncio.run(ComposioExecuteTool()(tool_slug="X", arguments={})).output
    assert "not configured" in out


def test_execute_success(monkeypatch):
    def h(method, url, body):
        assert url.endswith("/tools/execute")
        assert body["tool_slug"] == "GMAIL_SEND_EMAIL"
        return FakeResp(200, {"data": {"id": "msg1"}})

    out = run(ComposioExecuteTool(), monkeypatch, h, tool_slug="GMAIL_SEND_EMAIL", arguments={"to": "a@b.c"})
    assert "msg1" in out


def test_execute_401(monkeypatch):
    def h(method, url, body):
        return FakeResp(401, {})

    out = run(ComposioExecuteTool(), monkeypatch, h, tool_slug="X", arguments={})
    assert "re-paste" in out


def test_execute_403(monkeypatch):
    def h(method, url, body):
        return FakeResp(403, {})

    out = run(ComposioExecuteTool(), monkeypatch, h, tool_slug="X", arguments={})
    assert "re-paste" in out


def test_execute_400(monkeypatch):
    def h(method, url, body):
        return FakeResp(400, {"error": "bad args"})

    out = run(ComposioExecuteTool(), monkeypatch, h, tool_slug="X", arguments={})
    assert "Composio error 400" in out


def test_execute_passes_connected_account(monkeypatch):
    seen = {}

    def h(method, url, body):
        seen.update(body)
        return FakeResp(200, {"ok": True})

    run(ComposioExecuteTool(), monkeypatch, h, tool_slug="X", arguments={}, connected_account_id="acc1")
    assert seen["connected_account_id"] == "acc1"


def test_apps_lists_statuses(monkeypatch):
    def h(method, url, params):
        assert url.endswith("/connected_accounts")
        return FakeResp(200, {"items": [{"appName": "gmail", "status": "ACTIVE"}, {"app_name": "slack"}]})

    out = run(ComposioAppsTool(), monkeypatch, h)
    assert "- gmail: ACTIVE" in out
    assert "- slack: ?" in out


def test_list_tools_filters(monkeypatch):
    def h(method, url, params):
        assert params["app_names"] == "GMAIL"
        return FakeResp(
            200,
            {"items": [
                {"name": "GMAIL_SEND_EMAIL", "description": "Send an email"},
                {"name": "GMAIL_LIST_THREADS", "description": "List threads"},
            ]},
        )

    out = run(ComposioListToolsTool(), monkeypatch, h, app_name="gmail", search="send")
    assert "GMAIL_SEND_EMAIL" in out
    assert "GMAIL_LIST_THREADS" not in out
    out = run(ComposioListToolsTool(), monkeypatch, h, app_name="gmail")
    assert "GMAIL_LIST_THREADS" in out
