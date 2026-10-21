"""Web tools for the agent: fetch a URL as text, with a hardened SSRF guard.

The guard itself lives in app/ssrf.py (DNS-resolution screening, redirect
chain re-validation, credential/port blocking, body caps). This module only
wraps it as agent tools.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .ssrf import FetchFailed, SSRFBlocked, fetch_url_text
from .tool_base import BaseAgentTool, ToolResult


class WebFetchParams(BaseModel):
    url: str = Field(description="The http(s) URL to fetch and read as text")
    max_chars: int = Field(default=8000, description="Max characters of page text to return")


class WebFetchTool(BaseAgentTool[WebFetchParams]):
    name: str = "web_fetch"
    description: str = "Fetch a web page URL and return its text content. For reading docs, articles, and public pages."
    args_schema = WebFetchParams

    async def execute(self, params: WebFetchParams) -> ToolResult:
        try:
            result = fetch_url_text(params.url, max_chars=params.max_chars)
        except SSRFBlocked as e:
            return ToolResult(output=str(e))
        except FetchFailed as e:
            return ToolResult(output=f"Fetch failed: {e}")
        except Exception as e:
            return ToolResult(output=f"Fetch failed: {e}")
        return ToolResult(output=f"URL: {result.url}\nTitle: {result.title}\n\n{result.text}")


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
