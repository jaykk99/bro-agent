"""Web tools for the agent: fetch a URL as text, with an SSRF guard."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field
from uniharness.tools import BaseAgentTool
from uniharness.types import ToolResult


def _is_blocked_url(raw: str) -> str | None:
    """Returns a reason if the URL must not be fetched, else None."""
    try:
        u = urlparse(raw)
    except Exception:
        return "not a valid URL"
    if u.scheme not in ("http", "https"):
        return "only http(s) URLs are fetchable"
    host = (u.hostname or "").lower()
    if not host:
        return "no host"
    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".internal")
        or host in ("metadata.google.internal", "metadata.google.com")
    ):
        return "private/local hosts are not fetchable"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return "private IP ranges are not fetchable"
    except ValueError:
        pass  # not a literal IP — hostname already screened
    return None


def _html_to_text(html: str, max_chars: int = 8000) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, ch in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


class WebFetchParams(BaseModel):
    url: str = Field(description="The http(s) URL to fetch and read as text")
    max_chars: int = Field(default=8000, description="Max characters of page text to return")


class WebFetchTool(BaseAgentTool[WebFetchParams]):
    name: str = "web_fetch"
    description: str = "Fetch a web page URL and return its text content. For reading docs, articles, and public pages."
    args_schema = WebFetchParams

    async def execute(self, params: WebFetchParams) -> ToolResult:
        blocked = _is_blocked_url(params.url)
        if blocked:
            return ToolResult(output=f"Blocked: {blocked}")
        try:
            async with httpx.AsyncClient(timeout=25, follow_redirects=True, headers={"User-Agent": "bro-agent/1.0"}) as c:
                r = await c.get(params.url)
                if r.status_code >= 400:
                    return ToolResult(output=f"HTTP {r.status_code} fetching {params.url}")
                ctype = r.headers.get("content-type", "")
                if "html" in ctype:
                    text = _html_to_text(r.text, params.max_chars)
                else:
                    text = r.text[: params.max_chars]
                title = ""
                m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.I | re.S)
                if m:
                    title = m.group(1).strip()[:200]
                return ToolResult(output=f"URL: {params.url}\nTitle: {title}\n\n{text}")
        except Exception as e:
            return ToolResult(output=f"Fetch failed: {e}")


class WebSearchParams(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Max results to return")


class WebSearchTool(BaseAgentTool[WebSearchParams]):
    name: str = "web_search"
    description: str = "Search the web for current information. Returns titles, URLs, and snippets."
    args_schema = WebSearchParams

    async def execute(self, params: WebSearchParams) -> ToolResult:
        # DuckDuckGo HTML endpoint — no key needed.
        try:
            async with httpx.AsyncClient(timeout=25, headers={"User-Agent": "bro-agent/1.0"}) as c:
                r = await c.get("https://html.duckduckgo.com/html/", params={"q": params.query})
                if r.status_code >= 400:
                    return ToolResult(output=f"Search failed: HTTP {r.status_code}")
                out: list[str] = []
                for m in re.finditer(
                    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                    r.text,
                    re.S,
                ):
                    url, title, snippet = m.group(1), m.group(2), m.group(3)
                    title = re.sub(r"<[^>]+>", "", title).strip()
                    snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                    out.append(f"- {title}\n  {url}\n  {snippet}")
                    if len(out) >= params.max_results:
                        break
                return ToolResult(output="\n".join(out) if out else "No results found.")
        except Exception as e:
            return ToolResult(output=f"Search failed: {e}")


def web_tools() -> list[Any]:
    return [WebFetchTool(), WebSearchTool()]
