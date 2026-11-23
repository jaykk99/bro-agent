"""Pin the remaining untested paths in app/ssrf.py.

Coverage gaps left after test_ssrf.py: the empty-host screen, the
real _resolve_host path, the urlparse-failure branch, an unparseable
resolved IP, the generic DNS-check failure, and a redirect with no
Location header. No real network/DNS.
"""

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import ssrf  # noqa: E402
from app.ssrf import FetchFailed, check_url, fetch_url_text  # noqa: E402


def _public_dns(host: str) -> list[str]:
    return ["93.184.216.34"]


def test_empty_host_is_refused():
    assert check_url("http:///path") == "no host"


def test_resolve_host_returns_sorted_unique_ips():
    ips = ssrf._resolve_host("localhost")
    assert "127.0.0.1" in ips
    assert ips == sorted(set(ips))


def test_malformed_ipv6_url_is_refused():
    assert check_url("http://[::1") == "not a valid URL"


def test_unparseable_resolved_ip_is_refused(monkeypatch):
    monkeypatch.setattr(ssrf, "_resolve_host", lambda host: ["not-an-ip"])
    assert check_url("http://example.com/") == "could not parse resolved IP 'not-an-ip'"


def test_generic_dns_failure_is_refused(monkeypatch):
    def boom(host):
        raise RuntimeError("dns boom")

    monkeypatch.setattr(ssrf, "_resolve_host", boom)
    assert check_url("http://example.com/") == "DNS check failed"


def test_redirect_without_location_raises(monkeypatch):
    monkeypatch.setattr(ssrf, "_resolve_host", _public_dns)

    def handler(request):
        return httpx.Response(302, headers={})  # no Location header

    with pytest.raises(FetchFailed, match="redirect with no Location header"):
        fetch_url_text("http://example.com/", transport=httpx.MockTransport(handler))
