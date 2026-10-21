"""Composio tools for the agent — Jay's connected integrations.

Uses the Composio REST API with COMPOSIO_API_KEY from the environment.
Gracefully reports when the key is absent or invalid instead of failing.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .tool_base import BaseAgentTool, ToolResult

COMPOSIO_BASE = "https://backend.composio.dev/api/v3"


def _api_key() -> str:
    return os.environ.get("COMPOSIO_API_KEY", "").strip()


def _headers() -> dict[str, str]:
    return {"x-api-key": _api_key(), "Content-Type": "application/json"}


def _auth_error(status: int) -> str | None:
    if status == 401:
        return "Composio key is invalid (401) — Jay needs to re-paste it."
    if status == 403:
        return "Composio refused the key (403) — Jay needs to re-paste it."
    return None


class ComposioExecuteParams(BaseModel):
    tool_slug: str = Field(description="Composio tool slug, e.g. GMAIL_SEND_EMAIL or SLACK_SEND_MESSAGE")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool as a JSON object")
    connected_account_id: str = Field(default="", description="Optional connected account ID to run as")


class ComposioExecuteTool(BaseAgentTool[ComposioExecuteParams]):
    name: str = "composio_execute"
    description: str = (
        "Run an action on one of Jay's connected integrations via Composio "
        "(Gmail, Slack, GitHub, etc). Give the tool slug and its arguments. "
        "Use composio_list_tools first if you don't know the exact slug."
    )
    args_schema = ComposioExecuteParams

    async def execute(self, params: ComposioExecuteParams) -> ToolResult:
        key = _api_key()
        if not key:
            return ToolResult(output="Composio is not configured — COMPOSIO_API_KEY is missing.")
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                body: dict[str, Any] = {
                    "tool_slug": params.tool_slug,
                    "arguments": params.arguments,
                }
                if params.connected_account_id:
                    body["connected_account_id"] = params.connected_account_id
                r = await c.post(f"{COMPOSIO_BASE}/tools/execute", json=body, headers=_headers())
                auth = _auth_error(r.status_code)
                if auth:
                    return ToolResult(output=auth)
                data = r.json()
                if r.status_code >= 400:
                    return ToolResult(output=f"Composio error {r.status_code}: {str(data)[:500]}")
                return ToolResult(output=str(data)[:4000])
        except Exception as e:
            return ToolResult(output=f"Composio call failed: {e}")


class ComposioAppsParams(BaseModel):
    pass


class ComposioAppsTool(BaseAgentTool[ComposioAppsParams]):
    name: str = "composio_apps"
    description: str = "List Jay's connected Composio apps/integrations and their statuses."
    args_schema = ComposioAppsParams

    async def execute(self, params: ComposioAppsParams) -> ToolResult:
        key = _api_key()
        if not key:
            return ToolResult(output="Composio is not configured — COMPOSIO_API_KEY is missing.")
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(f"{COMPOSIO_BASE}/connected_accounts", headers=_headers())
                auth = _auth_error(r.status_code)
                if auth:
                    return ToolResult(output=auth)
                data = r.json()
                items = data.get("items", data) if isinstance(data, dict) else data
                lines = []
                for it in items[:30] if isinstance(items, list) else []:
                    app = it.get("appName") or it.get("app_name") or "?"
                    state = it.get("status") or it.get("state") or "?"
                    lines.append(f"- {app}: {state}")
                return ToolResult(output="\n".join(lines) if lines else str(data)[:2000])
        except Exception as e:
            return ToolResult(output=f"Composio call failed: {e}")


class ComposioListToolsParams(BaseModel):
    app_name: str = Field(description="App name to list tools for, e.g. GMAIL, SLACK, GITHUB")
    search: str = Field(default="", description="Optional keyword to filter tool names/descriptions")


class ComposioListToolsTool(BaseAgentTool[ComposioListToolsParams]):
    name: str = "composio_list_tools"
    description: str = (
        "List the available Composio tool slugs for one of Jay's connected apps, "
        "with short descriptions, so you can pick the right slug for composio_execute. "
        "Filter with `search` to narrow a big list."
    )
    args_schema = ComposioListToolsParams

    async def execute(self, params: ComposioListToolsTool) -> ToolResult:
        key = _api_key()
        if not key:
            return ToolResult(output="Composio is not configured — COMPOSIO_API_KEY is missing.")
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(
                    f"{COMPOSIO_BASE}/tools",
                    params={"app_names": params.app_name.upper()},
                    headers=_headers(),
                )
                auth = _auth_error(r.status_code)
                if auth:
                    return ToolResult(output=auth)
                if r.status_code >= 400:
                    return ToolResult(output=f"Composio error {r.status_code}: {r.text[:500]}")
                data = r.json()
                items = data.get("items", data) if isinstance(data, dict) else data
                if not isinstance(items, list):
                    return ToolResult(output=str(data)[:2000])
                needle = (params.search or "").lower()
                lines = []
                for it in items:
                    slug = it.get("name") or it.get("slug") or "?"
                    desc = (it.get("description") or "").strip().split("\n")[0][:120]
                    if needle and needle not in slug.lower() and needle not in desc.lower():
                        continue
                    lines.append(f"- {slug}: {desc}")
                    if len(lines) >= 40:
                        lines.append(f"... and {len(items) - 40} more (use `search` to narrow)")
                        break
                return ToolResult(output="\n".join(lines) if lines else f"No tools found for {params.app_name}.")
        except Exception as e:
            return ToolResult(output=f"Composio call failed: {e}")


def composio_tools() -> list[Any]:
    return [ComposioExecuteTool(), ComposioAppsTool(), ComposioListToolsTool()]
