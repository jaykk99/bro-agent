"""Pin the remaining untested paths in app/tools_composio.py.

Coverage gaps left after test_composio.py: the execute exception
branch, the apps/list-tools missing-key, auth, error-status and
exception branches, the list-tools non-list payload, the 40-item
truncation, and the composio_tools() registry. HTTP is faked.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.tools_composio as tc  # noqa: E402
from app.tools_composio import (  # noqa: E402
    ComposioAppsTool,
    ComposioExecuteTool,
    ComposioListToolsTool,
    composio_tools,
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
    return asyncio.run(tool(**kwargs)).output


def test_execute_exception_is_reported(monkeypatch):
    def h(method, url, body):
        raise RuntimeError("network down")

    out = run(ComposioExecuteTool(), monkeypatch, h, tool_slug="X", arguments={})
    assert "Composio call failed: network down" in out


def test_apps_no_key(monkeypatch):
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    out = asyncio.run(ComposioAppsTool()()).output
    assert "not configured" in out


def test_apps_auth_error(monkeypatch):
    def h(method, url, params):
        return FakeResp(403, {})

    out = run(ComposioAppsTool(), monkeypatch, h)
    assert "re-paste" in out


def test_apps_exception_is_reported(monkeypatch):
    def h(method, url, params):
        raise RuntimeError("dns down")

    out = run(ComposioAppsTool(), monkeypatch, h)
    assert "Composio call failed: dns down" in out


def test_apps_empty_items_falls_back_to_raw(monkeypatch):
    def h(method, url, params):
        return FakeResp(200, {"items": [], "meta": "none"})

    out = run(ComposioAppsTool(), monkeypatch, h)
    assert "none" in out


def test_list_tools_no_key(monkeypatch):
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    out = asyncio.run(ComposioListToolsTool()(app_name="GMAIL")).output
    assert "not configured" in out


def test_list_tools_auth_error(monkeypatch):
    def h(method, url, params):
        return FakeResp(401, {})

    out = run(ComposioListToolsTool(), monkeypatch, h, app_name="GMAIL")
    assert "re-paste" in out


def test_list_tools_error_status(monkeypatch):
    def h(method, url, params):
        return FakeResp(500, {"error": "boom"})

    out = run(ComposioListToolsTool(), monkeypatch, h, app_name="GMAIL")
    assert "Composio error 500" in out


def test_list_tools_non_list_items_falls_back_to_raw(monkeypatch):
    def h(method, url, params):
        return FakeResp(200, {"items": "not-a-list"})

    out = run(ComposioListToolsTool(), monkeypatch, h, app_name="GMAIL")
    assert "not-a-list" in out


def test_list_tools_truncates_after_40(monkeypatch):
    def h(method, url, params):
        return FakeResp(200, {"items": [
            {"name": f"TOOL_{i:02d}", "description": f"tool {i}"} for i in range(45)
        ]})

    out = run(ComposioListToolsTool(), monkeypatch, h, app_name="GMAIL")
    assert "TOOL_39" in out
    assert "TOOL_40" not in out
    assert "... and 5 more (use `search` to narrow)" in out


def test_list_tools_exception_is_reported(monkeypatch):
    def h(method, url, params):
        raise RuntimeError("timeout")

    out = run(ComposioListToolsTool(), monkeypatch, h, app_name="GMAIL")
    assert "Composio call failed: timeout" in out


def test_composio_tools_registry():
    assert [t.name for t in composio_tools()] == [
        "composio_execute",
        "composio_apps",
        "composio_list_tools",
    ]
