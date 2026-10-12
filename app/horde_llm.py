"""AI Horde chat model for bro-agent — keyless, genuinely uncensored models.

UniHarness agents require a tool-calling LLM. AI Horde workers (koboldcpp /
aphrodite serving Stheno, Heretic, Rocinante, etc.) do NOT support native
function calling, so this module emulates it the classic ReAct way:

  * bind_tools(tools) stores the tools and injects their schemas into the
    prompt as text.
  * The model is instructed to emit fenced ```tool_call JSON blocks.
  * _generate parses those blocks into AIMessage.tool_calls so LangGraph's
    ToolNode executes them like native tool calls.

Model chain for bro-agent: Venice (keyed, primary) -> Horde (keyless,
uncensored fallback). Never an aligned model.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

HORDE_API = "https://stablehorde.net/api/v2"
HORDE_ANON_KEY = "0000000000"

# Uncensored models, most-likely-online first. Horde live-filters this list:
# a job requesting only dead models sits in the queue forever, so passing
# several real options keeps it moving.
HORDE_UNCENSORED_MODELS = [
    "koboldcpp/L3-8B-Stheno-v3.2",
    "koboldcpp/Rocinante-X-12B",
    "koboldcpp/Magidonia-24B-v4.3",
    "koboldcpp/mini-magnum-12b-v1.1",
    "aphrodite/TheDrummer/Skyfall-31B-v4.2",
    "koboldcpp/gemma-4-31B-it-heretic",
]

TOOL_CALL_FENCE_RE = re.compile(r"```tool_call\s*\n?(.*?)```", re.DOTALL)
TOOL_CALL_JSON_RE = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)

TOOL_FORMAT_INSTRUCTIONS = """You have tools. Use them when they help; otherwise just reply.

To call a tool, output a fenced block EXACTLY like this, nothing else on those lines:
```tool_call
{"name": "tool_name", "arguments": {"arg1": "value1"}}
```
Rules:
- One tool per fenced block. You may output several blocks in one reply.
- "arguments" must be a JSON object matching the tool's args. Omit optional args you don't need.
- After tool results arrive, use them and continue. Never invent tool results.
- If no tool is needed, reply normally with no fenced block."""


def _tool_schema_text(tools: list[Any]) -> str:
    lines: list[str] = []
    for t in tools:
        name = getattr(t, "name", "?")
        desc = (getattr(t, "description", "") or "").strip().replace("\n", " ")
        args_text = ""
        try:
            schema = None
            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None:
                schema = args_schema.model_json_schema() if hasattr(args_schema, "model_json_schema") else None
            elif isinstance(getattr(t, "args", None), dict):
                props = t.args
                schema = {"properties": props}
            if schema:
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parts = []
                for pname, pspec in props.items():
                    ptype = pspec.get("type", "string") if isinstance(pspec, dict) else "string"
                    pdesc = (pspec.get("description", "") or "") if isinstance(pspec, dict) else ""
                    star = "*" if pname in required else ""
                    parts.append(f'{pname}{star}: {ptype}{" - " + pdesc if pdesc else ""}')
                args_text = "; ".join(parts)
        except Exception:
            args_text = ""
        lines.append(f"- {name}: {desc}" + (f"\n  args: {args_text}" if args_text else ""))
    return "\n".join(lines)


def _messages_to_prompt(messages: list[BaseMessage], tools_text: str) -> str:
    """Flatten LangChain messages into one instruction-style prompt for Horde."""
    system_parts: list[str] = []
    convo: list[str] = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        if isinstance(m, SystemMessage):
            system_parts.append(content)
        elif isinstance(m, HumanMessage):
            convo.append(f"User: {content}")
        elif isinstance(m, AIMessage):
            # Replay prior tool calls as text so the model sees what happened.
            tc_text = ""
            for tc in m.tool_calls or []:
                tc_text += f'\n```tool_call\n{{"name": "{tc["name"]}", "arguments": {json.dumps(tc["args"])}}}\n```'
            convo.append(f"Assistant: {content}{tc_text}".rstrip())
        elif isinstance(m, ToolMessage):
            convo.append(f"Tool result ({m.name}): {content}")
        else:
            convo.append(f"{m.type}: {content}")
    system = "\n\n".join(system_parts).strip()
    prompt = ""
    if system:
        prompt += system + "\n\n"
    if tools_text:
        prompt += "Available tools:\n" + tools_text + "\n\n" + TOOL_FORMAT_INSTRUCTIONS + "\n\n"
    prompt += "\n".join(convo)
    prompt += "\nAssistant:"
    return prompt


def _parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split model output into (clean_text, tool_calls)."""
    calls: list[dict] = []
    for m in TOOL_CALL_FENCE_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        name = obj.get("name")
        args = obj.get("arguments", {}) or {}
        if name and isinstance(args, dict):
            calls.append({"name": name, "args": args})
    clean = TOOL_CALL_FENCE_RE.sub("", text).strip()
    # Lenient fallback: bare {"name": "...", ...} JSON without the fence.
    if not calls:
        for m in TOOL_CALL_JSON_RE.finditer(text):
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            name = obj.get("name")
            args = {k: v for k, v in obj.items() if k != "name"}
            if name:
                calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
                clean = text.replace(m.group(0), "").strip()
                break
    return clean, calls


def _horde_generate(prompt: str, models: list[str], max_context_length: int,
                    max_length: int, temperature: float, timeout_s: int) -> str:
    """Submit a prompt to AI Horde and poll until done. Raises on failure."""
    headers = {"apikey": HORDE_ANON_KEY, "Content-Type": "application/json"}
    submit_body = {
        "prompt": prompt,
        "params": {
            "max_context_length": max_context_length,
            "max_length": max_length,
            "temperature": temperature,
        },
        "models": models,
    }
    with httpx.Client(timeout=30) as client:
        sub = client.post(f"{HORDE_API}/generate/text/async", headers=headers, json=submit_body)
        if sub.status_code == 429:
            raise RuntimeError("horde: rate limited on submit (429)")
        sub.raise_for_status()
        job_id = sub.json().get("id")
        if not job_id:
            raise RuntimeError("horde: no job id returned")

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(4)
            chk = client.get(f"{HORDE_API}/generate/text/status/{job_id}", headers={"apikey": HORDE_ANON_KEY})
            if chk.status_code != 200:
                continue
            s = chk.json()
            if s.get("faulted"):
                raise RuntimeError("horde: job faulted")
            if s.get("done"):
                gens = s.get("generations") or []
                if gens and gens[0].get("text"):
                    return gens[0]["text"]
                raise RuntimeError("horde: done but empty generation")
    raise RuntimeError("horde: timed out waiting for generation")


class HordeChatModel(BaseChatModel):
    """LangChain chat model backed by AI Horde's keyless uncensored workers,
    with prompt-based tool-call emulation so UniHarness agents can use it."""

    horde_models: list[str] = HORDE_UNCENSORED_MODELS
    max_context_length: int = 8192
    max_length: int = 1500
    temperature: float = 0.3
    timeout_s: int = 240

    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "horde-uncensored"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN202
        new = self.model_copy(deep=False)
        new._bound_tools = list(tools)
        return new

    def _build_prompt(self, messages: list[BaseMessage]) -> str:
        tools_text = _tool_schema_text(self._bound_tools) if self._bound_tools else ""
        prompt = _messages_to_prompt(messages, tools_text)
        # Keep the prompt inside the worker's context window.
        max_chars = self.max_context_length * 3  # ~3 chars per token, conservative
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\nAssistant:"
        return prompt

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._build_prompt(messages)
        logger.info("horde: submitting prompt (%d chars)", len(prompt))
        text = _horde_generate(
            prompt,
            models=self.horde_models,
            max_context_length=self.max_context_length,
            max_length=self.max_length,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
        )
        clean, calls = _parse_tool_calls(text)
        tool_calls = [
            {"name": c["name"], "args": c["args"], "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
            for c in calls
        ]
        logger.info("horde: got %d chars, %d tool calls", len(text), len(tool_calls))
        message = AIMessage(content=clean, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])
