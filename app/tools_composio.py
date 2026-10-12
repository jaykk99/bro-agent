"""Composio tools for the agent — Jay's connected integrations.

Uses the Composio REST API with COMPOSIO_API_KEY from the environment.
Gracefully reports when the key is absent or invalid instead of failing.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field
from uniharness.tools import BaseAgentTool
from uniharness.types import ToolResult

COMPOSIO_BASE = "https://backend.composio.dev/api/v3"


def _api_key() -> str:
    return os.environ.get("COMPOSIO_API_KEY", "").strip()


class ComposioExecuteParams(BaseModel):
    tool_slug: str = Field(description="Composio tool slug, e.g. GMAIL_SEND_EMAIL or SLACK_SEND_MESSAGE")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool as a JSON object")
    connected_account_id: str = Field(default="", description="Optional connected account ID to run as")


class ComposioExecuteTool(BaseAgentTool[ComposioExecuteParams]):
    name: str = "composio_execute"
    description: str = (
        "Run an action on one of Jay's connected integrations via Composio "
        "(Gmail, Slack, GitHub, etc). Give the tool slug and its arguments."
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
                r = await c.post(
                    f"{COMPOSIO_BASE}/tools/execute",
                    json=body,
                    headers={"x-api-key": key, "Content-Type": "application/json"},
                )
                if r.status_code == 401:
                    return ToolResult(output="Composio key is invalid (401) — Jay needs to re-paste it.")
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
                r = await c.get(
                    f"{COMPOSIO_BASE}/connected_accounts",
                    headers={"x-api-key": key},
                )
                if r.status_code == 401:
                    return ToolResult(output="Composio key is invalid (401) — Jay needs to re-paste it.")
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


def composio_tools() -> list[Any]:
    return [ComposioExecuteTool(), ComposioAppsTool()]
