"""SSRF guard for bro-agent's outbound web fetches.

The old check (in app/tools_web.py) only screened the literal hostname —
a hostname that *resolves* to a private IP, or a redirect chain that lands
on one, sailed straight through. This module closes those holes:

  * literal-IP screening (loopback / private / link-local / reserved /
    multicast, IPv4 + IPv6)
  * DNS resolution screening — every resolved A/AAAA record for the host
    must be public, otherwise the fetch is refused
  * manual redirect handling — each hop in the chain is re-validated
    (URL + DNS), max 5 hops, relative Location headers resolved properly
  * credentials in the URL (user:pass@host) are refused
  * non-standard ports are refused (only 80/443 or scheme-default)
  * response body capped (default 2 MB) before text extraction

Pure-stdlib + httpx. Test seam: `fetch_url_text(..., transport=...)`
accepts an httpx transport (MockTransport in tests), and `_resolve_host`
is monkeypatchable for DNS fixtures.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

USER_AGENT = "bro-agent/1.0"

MAX_REDIRECTS = 5
MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MB cap before text extraction

# Hostnames that are always off-limits, even if DNS would resolve them
# to something public (defense in depth — some resolvers lie).
_ALWAYS_BLOCKED_HOSTS = {
    "metadata.google.internal",
    "metadata.google.com",  # GCP metadata service
    "169.254.169.254",  # AWS / cloud metadata IP
    "metadata",  # some internal resolvers answer this
}

# Suffixes that indicate non-routable / internal names.
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".corp", ".home", ".intranet")

_ALLOWED_PORTS = {80, 443}


class SSRFBlocked(Exception):
    """Raised when a URL is refused by the SSRF guard."""


class FetchFailed(Exception):
    """Raised when the fetch itself fails (network, HTTP error, oversize)."""


@dataclass
class FetchResult:
    url: str  # final URL after redirects
    title: str
    text: str
    status_code: int


def _is_bad_ip(ip: ipaddress._BaseAddress) -> bool:
    """True for any non-public-unicast address."""
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _screen_literal_host(host: str) -> Optional[str]:
    """Refuse obviously-bad hostnames / literal IPs. Returns reason or None."""
    h = (host or "").lower().rstrip(".")
    if not h:
        return "no host"
    if h in _ALWAYS_BLOCKED_HOSTS:
        return "cloud metadata / internal host"
    if h == "localhost" or h.endswith(_BLOCKED_SUFFIXES):
        return "private/local host"
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return None  # not a literal IP — DNS screening handles it
    if _is_bad_ip(ip):
        return "private IP range"
    return None


def _resolve_host(host: str) -> list[str]:
    """Resolve a hostname to its IP strings. Raises socket.gaierror on failure."""
    infos = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def check_url(raw: str) -> Optional[str]:
    """Full pre-flight check: scheme, credentials, port, literal host, DNS.

    Returns a human-readable reason when the URL must NOT be fetched,
    else None. Never raises.
    """
    try:
        u = urlparse(raw)
    except Exception:
        return "not a valid URL"
    if u.scheme not in ("http", "https"):
        return "only http(s) URLs are fetchable"
    if u.username or u.password:
        return "URLs with credentials are not fetchable"
    host = (u.hostname or "").lower()
    reason = _screen_literal_host(host)
    if reason:
        return reason
    port = u.port or (443 if u.scheme == "https" else 80)
    if port not in _ALLOWED_PORTS:
        return f"port {port} is not fetchable (only 80/443)"
    # DNS check — the hole the old guard missed: a public-looking hostname
    # that resolves to a private IP (DNS rebinding / evil DNS).
    try:
        for ip_str in _resolve_host(host):
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return f"could not parse resolved IP {ip_str!r}"
            if _is_bad_ip(ip):
                return f"host resolves to non-public IP {ip_str}"
    except (socket.gaierror, UnicodeError):
        return "host does not resolve"
    except Exception:
        return "DNS check failed"
    return None


def _html_to_text(html: str, max_chars: int) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, ch in [
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ]:
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def fetch_url_text(
    raw_url: str,
    max_chars: int = 8000,
    timeout_s: int = 25,
    transport: Optional[httpx.BaseTransport] = None,
) -> FetchResult:
    """Fetch a URL as text, with full SSRF protection including the redirect chain.

    Raises SSRFBlocked when any hop is refused, FetchFailed on network/HTTP
    errors. Redirects are followed manually (max 5) so every hop is
    re-validated — httpx's follow_redirects would happily land on a private IP.
    """
    reason = check_url(raw_url)
    if reason:
        raise SSRFBlocked(f"Blocked: {reason}")

    client_kwargs: dict = {"timeout": timeout_s, "headers": {"User-Agent": USER_AGENT}}
    if transport is not None:
        client_kwargs["transport"] = transport

    url = raw_url
    with httpx.Client(follow_redirects=False, **client_kwargs) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            resp = client.get(url)
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location", "")
                if not location:
                    raise FetchFailed(f"redirect with no Location header from {url}")
                nxt = urljoin(url, location)
                hop_reason = check_url(nxt)
                if hop_reason:
                    raise SSRFBlocked(f"Blocked redirect target {nxt}: {hop_reason}")
                url = nxt
                continue
            if resp.status_code >= 400:
                raise FetchFailed(f"HTTP {resp.status_code} fetching {url}")
            body = resp.content
            if len(body) > MAX_BODY_BYTES:
                raise FetchFailed(f"response too large ({len(body)} bytes, cap {MAX_BODY_BYTES})")
            ctype = resp.headers.get("content-type", "")
            raw_text = body.decode(resp.encoding or "utf-8", errors="replace")
            title = ""
            m = re.search(r"<title[^>]*>(.*?)</title>", raw_text, re.I | re.S)
            if m:
                title = re.sub(r"\s+", " ", m.group(1)).strip()[:200]
            text = _html_to_text(raw_text, max_chars) if "html" in ctype else raw_text[:max_chars]
            return FetchResult(url=url, title=title, text=text, status_code=resp.status_code)
        raise FetchFailed(f"too many redirects (>{MAX_REDIRECTS}) starting from {raw_url}")
