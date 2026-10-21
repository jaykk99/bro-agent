"""Tests for the hardened SSRF guard (app/ssrf.py). No real network/DNS."""

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import ssrf  # noqa: E402
from app.ssrf import FetchFailed, SSRFBlocked, check_url, fetch_url_text  # noqa: E402


# Public-IP fixture: every hostname resolves to a public address unless
# a test overrides it via monkeypatch.
def _public_dns(host: str) -> list[str]:
    return ["93.184.216.34"]  # example.com


@pytest.fixture(autouse=True)
def _dns(monkeypatch):
    monkeypatch.setattr(ssrf, "_resolve_host", _public_dns)


def test_literal_private_ips_blocked():
    for url in [
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://172.16.9.9:80/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[::1]/",
        "http://[fc00::1]/",
    ]:
        assert check_url(url) is not None, url


def test_local_hostnames_blocked():
    for url in [
        "http://localhost:8000/",
        "http://foo.localhost/x",
        "http://printer.local/",
        "http://svc.internal/api",
        "http://metadata.google.internal/",
    ]:
        assert check_url(url) is not None, url


def test_non_http_schemes_blocked():
    assert check_url("file:///etc/passwd") is not None
    assert check_url("ftp://example.com/x") is not None
    assert check_url("gopher://example.com/") is not None


def test_credentials_in_url_blocked():
    assert check_url("http://user:pass@example.com/") is not None


def test_weird_ports_blocked():
    assert check_url("http://example.com:22/") is not None
    assert check_url("http://example.com:8080/") is not None
    # default / standard ports are fine
    assert check_url("http://example.com/") is None
    assert check_url("https://example.com:443/") is None


def test_dns_rebinding_host_blocked(monkeypatch):
    # The hole the old guard missed: public-looking hostname that
    # resolves to a private IP.
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["127.0.0.1"])
    assert check_url("http://evil.example.com/") is not None


def test_dns_rebinding_ipv6_blocked(monkeypatch):
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["::1"])
    assert check_url("http://evil.example.com/") is not None


def test_unresolvable_host_blocked(monkeypatch):
    import socket

    def _boom(host: str) -> list[str]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(ssrf, "_resolve_host", _boom)
    assert check_url("http://does-not-exist-xyz.example/") is not None


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_fetch_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><head><title>Hi</title></head><body><p>hello world</p></body></html>",
        )

    r = fetch_url_text("http://example.com/", transport=_transport(handler))
    assert r.title == "Hi"
    assert "hello world" in r.text
    assert r.url == "http://example.com/"


def test_fetch_follows_redirects_revalidating_each_hop():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url) == "http://example.com/a":
            return httpx.Response(302, headers={"location": "/b"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="final")

    r = fetch_url_text("http://example.com/a", transport=_transport(handler))
    assert r.text == "final"
    assert r.url == "http://example.com/b"
    assert seen == ["http://example.com/a", "http://example.com/b"]


def test_redirect_to_private_ip_blocked(monkeypatch):
    # evil hop resolves to 10.x — the redirect target must be refused.
    calls = {"n": 0}

    def dns(host: str) -> list[str]:
        calls["n"] += 1
        return ["10.9.9.9"] if calls["n"] > 1 else ["93.184.216.34"]

    monkeypatch.setattr(ssrf, "_resolve_host", dns)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.example/secret"})

    with pytest.raises(SSRFBlocked):
        fetch_url_text("http://example.com/start", transport=_transport(handler))


def test_redirect_loop_capped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://example.com/loop"})

    with pytest.raises(FetchFailed, match="too many redirects"):
        fetch_url_text("http://example.com/loop", transport=_transport(handler))


def test_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    with pytest.raises(FetchFailed, match="HTTP 404"):
        fetch_url_text("http://example.com/missing", transport=_transport(handler))


def test_oversize_body_rejected():
    big = "x" * (ssrf.MAX_BODY_BYTES + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text=big)

    with pytest.raises(FetchFailed, match="too large"):
        fetch_url_text("http://example.com/big", transport=_transport(handler))


def test_blocked_initial_url_raises_ssrf_blocked():
    with pytest.raises(SSRFBlocked):
        fetch_url_text("http://127.0.0.1/", transport=_transport(lambda r: httpx.Response(200)))


def test_script_and_style_stripped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><script>alert(1)</script><style>.x{}</style><p>keep me</p></body></html>",
        )

    r = fetch_url_text("http://example.com/", transport=_transport(handler))
    assert "alert" not in r.text
    assert "keep me" in r.text
