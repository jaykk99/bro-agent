"""CLI helper tests — rename/delete/export hit the right endpoints."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.cli as cli  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttpx:
    """Records calls, returns canned responses. Assigned over cli.httpx."""

    def __init__(self, responses: dict):
        self.responses = responses  # (method, url_prefix) -> FakeResponse
        self.calls = []

    def _call(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        for (m, prefix), resp in self.responses.items():
            if m == method and url.startswith(prefix):
                return resp
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kwargs):
        return self._call("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._call("POST", url, **kwargs)

    def delete(self, url, **kwargs):
        return self._call("DELETE", url, **kwargs)


def _install(monkeypatch, responses):
    fake = FakeHttpx(responses)
    monkeypatch.setattr(cli, "httpx", fake)
    monkeypatch.setattr(cli, "BASE", "http://test")
    monkeypatch.setattr(cli, "CODE", "secret")
    return fake


def test_rename_hits_endpoint(monkeypatch):
    fake = _install(monkeypatch, {("POST", "http://test/api/threads/abc/rename"): FakeResponse(200)})
    assert cli.rename_thread("abc", "  new title  ") is True
    method, url, kwargs = fake.calls[0]
    assert method == "POST" and url.endswith("/api/threads/abc/rename")
    assert kwargs["json"] == {"title": "new title", "access_code": "secret"}


def test_rename_rejects_empty():
    assert cli.rename_thread("", "x") is False
    assert cli.rename_thread("abc", "   ") is False


def test_rename_failure_returns_false(monkeypatch):
    _install(monkeypatch, {("POST", "http://test/api/threads/abc/rename"): FakeResponse(404)})
    assert cli.rename_thread("abc", "x") is False


def test_delete_hits_endpoint(monkeypatch):
    fake = _install(monkeypatch, {("DELETE", "http://test/api/threads/abc"): FakeResponse(200)})
    assert cli.delete_thread("abc") is True
    method, url, kwargs = fake.calls[0]
    assert method == "DELETE" and url.endswith("/api/threads/abc")
    assert kwargs["params"] == {"access_code": "secret"}


def test_delete_rejects_empty():
    assert cli.delete_thread("") is False


def test_export_writes_file(monkeypatch, tmp_path):
    body = b"# bro chat export\n\n## Jay\n\nhi\n"
    fake = _install(
        monkeypatch,
        {("GET", "http://test/api/threads/abc/export"): FakeResponse(200, content=body)},
    )
    dest = str(tmp_path / "out.md")
    path = cli.export_thread("abc", dest)
    assert path == dest
    assert open(dest, "rb").read() == body
    method, url, kwargs = fake.calls[0]
    assert method == "GET" and url.endswith("/api/threads/abc/export")


def test_export_requires_thread():
    try:
        cli.export_thread("")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
