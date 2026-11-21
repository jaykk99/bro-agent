"""AI Horde chat model for bro-agent — keyless, genuinely uncensored models.

UniHarness agents require a tool-calling LLM. AI Horde workers (koboldcpp /
aphrodite serving Stheno, Heretic, Rocinante, etc.) do NOT support native
function calling, so this module emulates it the classic ReAct way:

  * bind_tools(tools) stores the tools and injects their schemas into the
    prompt as text.
  * The model is instructed to emit fenced ```tool_call JSON blocks.
  * _generate parses those blocks into AIMessage.tool_calls so LangGraph's
    ToolNode executes them like native tool calls.

Model chain for bro-agent: Horde (keyless, uncensored primary) ->
Pollinations (keyless availability fallback). Never an aligned model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)

HORDE_APIS = [
    "https://stablehorde.net/api/v2",
    "https://aihorde.net/api/v2",
]
HORDE_ANON_KEY = "0000000000"
HORDE_USER_AGENT = "bro-agent/1.0 (https://github.com/jaykk99/bro-agent)"


def _horde_key() -> str:
    """Anon key works keyless; a registered free Horde key (HORDE_API_KEY)
    gets better queue priority and is less likely to be filtered."""
    return os.environ.get("HORDE_API_KEY", "").strip() or HORDE_ANON_KEY

# Uncensored models, most-workers-online first (verified 2026-10-15).
# Horde live-filters this list: a job requesting only dead models sits in
# the queue forever, so passing several real options keeps it moving.
HORDE_UNCENSORED_MODELS = [
    "koboldcpp/L3-8B-Stheno-v3.2",
    "koboldcpp/Rocinante-X-12B-v1",
    "koboldcpp/gemma-4-31B-it-heretic",
    "koboldcpp/gemma-4-26B-A4B-it-heretic",
    "koboldcpp/Gemma-4-E4B-it-Ultra-Uncensored-Heretic",
    "koboldcpp/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M",
]

# Live model refresh: the hardcoded list above goes stale as workers come
# and go. resolve_models() asks the Horde API which uncensored models have
# workers online right now (cached 6h), falling back to the hardcoded list
# when the API is unreachable.
MODELS_CACHE_S = 6 * 3600
# Status-poll fail-fast: consecutive non-200 (non-403) poll responses
# before the host is declared sick and the call fails over (see
# _horde_generate_on / _ahorde_generate_on).
POLL_ERROR_THRESHOLD = 3
_UNCENSORED_HINTS = ("stheno", "heretic", "rocinante", "uncensored")
_models_cache: tuple[float, list[str]] | None = None


def invalidate_models_cache() -> None:
    """Drop the cached live-model list so the next call re-refreshes it.

    Called when a Horde job times out: a timeout usually means the
    requested models are dead or stuck in the queue, and reusing the
    same cached list for up to 6h would repeat the failure on every
    turn. The next resolve_models() call re-asks the live API.
    """
    global _models_cache
    _models_cache = None


# Poll pacing: a job deep in the queue (queue_position > 0) is not going
# to finish in the next 2s, so polling every 4s wastes requests on a
# community API; a job already running (queue_position 0) finishes soon,
# so poll faster to return the answer sooner. Unknown position keeps 4s.
POLL_INTERVAL_RUNNING_S = 2.0
POLL_INTERVAL_DEFAULT_S = 4.0
POLL_INTERVAL_QUEUED_S = 6.0


def _poll_interval_s(status: dict | None) -> float:
    """Seconds to wait before the next status poll, from the last payload."""
    if not isinstance(status, dict):
        return POLL_INTERVAL_DEFAULT_S
    qp = status.get("queue_position")
    if qp is None:
        return POLL_INTERVAL_DEFAULT_S
    try:
        qp = int(qp)
    except (TypeError, ValueError):
        return POLL_INTERVAL_DEFAULT_S
    return POLL_INTERVAL_RUNNING_S if qp <= 0 else POLL_INTERVAL_QUEUED_S


def _retry_after_s(resp: Any, cap_s: float = 10.0) -> float | None:
    """Seconds to wait from a 429's Retry-After header, capped.

    Returns None when the header is missing or unparseable — the caller
    then fails over immediately instead of guessing a wait. A huge
    Retry-After (an edge telling us to go away for minutes) is capped:
    the router's job is to fail over to the next host/provider, not to
    sit out a long ban on one host.
    """
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return min(seconds, cap_s)


def _check_submit_response(resp: Any) -> None:
    """Raise for a submit response that cannot proceed (403/429/other error)."""
    if resp.status_code == 403:
        raise RuntimeError("horde: 403 forbidden on submit (edge blocked this network)")
    if resp.status_code == 429:
        raise RuntimeError("horde: rate limited on submit (429)")
    resp.raise_for_status()


def _fetch_live_models(http_get=None) -> list[str]:
    """Query the Horde model list; return online uncensored model names.

    Tries each known Horde API host in turn: some networks get 403s from
    one host's edge (Cloudflare IP filtering) while the other works, and
    a refresh that only ever asked the blocked host silently fell back
    to the hardcoded list every 6h even though the live list was one
    host away. Raises the last error when every host fails, so the
    caller falls back to the hardcoded list; a host that answers but
    lists no online uncensored models is tried next too, and an empty
    result from all hosts returns [] (the caller treats that as no
    refresh, same as before).
    """
    get = http_get or httpx.get
    last_error: Exception | None = None
    for base in ("https://stablehorde.net", "https://aihorde.net"):
        try:
            r = get(
                f"{base}/api/v2/status/models",
                headers={"User-Agent": HORDE_USER_AGENT},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            last_error = e
            logger.debug("horde: live model refresh failed on %s: %s", base, e)
            continue
        items = data if isinstance(data, list) else []
        live: list[tuple[int, str]] = []
        for it in items:
            name = str(it.get("name", ""))
            workers = int(it.get("count", 0) or 0)
            if workers <= 0:
                continue
            lname = name.lower()
            if any(h in lname for h in _UNCENSORED_HINTS):
                # prefer koboldcpp-served models (text gen), then most workers
                prio = 0 if lname.startswith("koboldcpp/") else 1
                live.append((prio, -workers, name))
        live.sort()
        if live:
            return [name for _, _, name in live]
        logger.debug("horde: %s listed no online uncensored models", base)
    if last_error is not None:
        raise last_error
    return []


def resolve_models(configured: list[str], http_get=None) -> list[str]:
    """Resolve which Horde models to request.

    When the caller uses the default list, refresh it against the live
    Horde API (6h cache) so jobs don't queue forever on dead models.
    Custom lists are used as-is. Never returns empty — falls back to the
    hardcoded list when the API fails.
    """
    global _models_cache
    if configured != HORDE_UNCENSORED_MODELS:
        return configured
    now = time.time()
    if _models_cache and now - _models_cache[0] < MODELS_CACHE_S:
        return _models_cache[1]
    try:
        live = _fetch_live_models(http_get)
        if live:
            _models_cache = (now, live)
            logger.info("horde: refreshed model list (%d live uncensored)", len(live))
            return live
    except Exception as e:
        logger.warning("horde: live model refresh failed, using defaults: %s", e)
    _models_cache = (now, HORDE_UNCENSORED_MODELS)
    return HORDE_UNCENSORED_MODELS

TOOL_CALL_FENCE_RE = re.compile(r"```tool_call\s*\n?(.*?)```", re.DOTALL)

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


def _messages_to_parts(messages: list[BaseMessage], tools_text: str) -> tuple[str, list[str]]:
    """Split the flattened prompt into (header, convo_blocks).

    Header = system prompt + available-tools block + format instructions;
    convo_blocks = one block per flattened message turn. The split exists
    so _build_prompt can trim oldest-first: the header is what the model
    needs to follow instructions, the newest convo blocks are what it
    needs to answer.
    """
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
    header = ""
    if system:
        header += system + "\n\n"
    if tools_text:
        header += "Available tools:\n" + tools_text + "\n\n" + TOOL_FORMAT_INSTRUCTIONS + "\n\n"
    return header, convo


def _messages_to_prompt(messages: list[BaseMessage], tools_text: str) -> str:
    """Flatten LangChain messages into one instruction-style prompt for Horde."""
    header, convo = _messages_to_parts(messages, tools_text)
    return header + "\n".join(convo) + "\nAssistant:"


def _trim_marker(dropped: int) -> str:
    return f"[{dropped} older message(s) trimmed to fit context]"


def _fit_prompt(header: str, convo: list[str], max_chars: int) -> str:
    """Fit a prompt inside max_chars, trimming oldest convo first.

    The header (system prompt + tool instructions) is never dropped and
    the newest convo block is always kept — if that single block alone
    is still too long, its tail (the most recent content) is kept. A
    marker notes how many older blocks were trimmed so the model knows
    context is missing instead of hallucinating continuity. The result
    always ends with the Assistant cue and never exceeds max_chars.
    """
    tail = "\nAssistant:"
    max_chars = max(max_chars, len(tail) + 1)
    full = header + "\n".join(convo) + tail
    if len(full) <= max_chars:
        return full
    if len(header) + len(tail) >= max_chars:
        # Even the instructions alone overflow: keep the newest content.
        return full[-(max_chars - len(tail)):] + tail
    kept = list(convo)
    dropped = 0
    while len(kept) > 1:
        marker = _trim_marker(dropped + 1)
        if len(header) + len(marker) + 1 + len("\n".join(kept[1:])) + len(tail) <= max_chars:
            dropped += 1
            kept = kept[1:]
            break
        dropped += 1
        kept = kept[1:]
    body = "\n".join(kept)
    if dropped:
        body = _trim_marker(dropped) + "\n" + body
    if len(header) + len(body) + len(tail) > max_chars:
        # One huge newest block: keep its tail.
        keep = max_chars - len(header) - len(tail)
        body = body[-keep:] if keep > 0 else ""
    result = header + body + tail
    if len(result) > max_chars:  # last-resort guard, never over budget
        result = result[-(max_chars - len(tail)):] + tail
    return result


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
    # raw_decode (not a [^{}]* regex) so nested "arguments" objects parse
    # too — the old regex could not span nested braces, so the most
    # natural bare form (same shape as the fenced format) was silently
    # dropped and the raw JSON shown to the user as reply text instead
    # of executing the tool call.
    if not calls:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, end = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            if not name or not isinstance(name, str):
                continue
            args = obj.get("arguments")
            if not isinstance(args, dict):
                # Flat form: every key except "name" is an argument.
                args = {k: v for k, v in obj.items() if k != "name"}
            calls.append({"name": name, "args": args})
            clean = (text[:i] + text[i + end:]).strip()
            break
    return clean, calls


def _submit_headers(api_key: str) -> dict:
    return {
        "apikey": api_key,
        "Content-Type": "application/json",
        "User-Agent": HORDE_USER_AGENT,
    }


def _submit_body(prompt: str, models: list[str], max_context_length: int,
                 max_length: int, temperature: float, stop: list[str] | None = None) -> dict:
    params: dict[str, Any] = {
        "max_context_length": max_context_length,
        "max_length": max_length,
        "temperature": temperature,
    }
    if stop:
        # Horde's param name is stop_sequence (koboldcpp-style); the
        # LangChain `stop` list maps straight onto it.
        params["stop_sequence"] = list(stop)
    return {
        "prompt": prompt,
        "params": params,
        "models": models,
    }


def _horde_generate(prompt: str, models: list[str], max_context_length: int,
                    max_length: int, temperature: float, timeout_s: int,
                    stop: list[str] | None = None) -> str:
    """Submit a prompt to AI Horde and poll until done. Raises on failure.

    Tries each known Horde API host in turn — some networks get 403s from
    one host's edge (Cloudflare IP filtering) while the other works."""
    api_key = _horde_key()
    headers = _submit_headers(api_key)
    body = _submit_body(prompt, models, max_context_length, max_length, temperature, stop=stop)
    last_error: Exception | None = None
    for api in HORDE_APIS:
        try:
            return _horde_generate_on(api, headers, body, timeout_s)
        except Exception as e:
            # Broad on purpose: a 5xx raises httpx.HTTPStatusError and a
            # non-JSON reply raises ValueError, and neither is a
            # RuntimeError — catching only RuntimeError aborted the whole
            # call on the first host instead of failing over to the next.
            logger.warning("horde: %s failed: %s", api, e)
            last_error = e
            continue
    raise RuntimeError(f"horde: all endpoints failed (last: {last_error})")


async def _ahorde_generate(prompt: str, models: list[str], max_context_length: int,
                           max_length: int, temperature: float, timeout_s: int,
                           stop: list[str] | None = None) -> str:
    """Async twin of _horde_generate — same failover, non-blocking polls.

    The sync version sleeps the calling thread between polls; inside the
    FastAPI event loop that stalls every other request for up to the
    Horde timeout (60s by default, see HORDE_CHAT_TIMEOUT_S), so
    the agent path prefers this one.
    """
    api_key = _horde_key()
    headers = _submit_headers(api_key)
    body = _submit_body(prompt, models, max_context_length, max_length, temperature, stop=stop)
    last_error: Exception | None = None
    for api in HORDE_APIS:
        try:
            return await _ahorde_generate_on(api, headers, body, timeout_s)
        except Exception as e:
            # Broad on purpose — see _horde_generate: a 5xx or a non-JSON
            # reply from one host must fail over, not abort the call.
            logger.warning("horde: %s failed: %s", api, e)
            last_error = e
            continue
    raise RuntimeError(f"horde: all endpoints failed (last: {last_error})")


def _parse_status_payload(s: dict) -> str | None:
    """Returns the generation text when the job is done, else None. Raises on fault."""
    if s.get("faulted"):
        raise RuntimeError("horde: job faulted")
    if s.get("done"):
        gens = s.get("generations") or []
        if gens and gens[0].get("text"):
            return gens[0]["text"]
        raise RuntimeError("horde: done but empty generation")
    return None


def _log_queue_progress(job_id: str, s: dict) -> None:
    """Debug-log where the job sits in the Horde queue, when the payload says.

    Queue depth is the only signal for whether a slow job is moving or
    stuck — without it a 60s wait is a black box in the logs.
    """
    qp = s.get("queue_position")
    wt = s.get("wait_time")
    if qp is None and wt is None:
        return
    logger.debug("horde: job %s queue_position=%s wait_time=%s", job_id, qp, wt)


def _cancel_job(client, api: str, headers: dict, job_id: str) -> None:
    """Best-effort DELETE of a Horde job nobody waits for any more.

    A job that timed out, hit a sick status endpoint, or was refused by
    the edge would otherwise stay queued/running on a community worker
    until Horde reaps it — occupying capacity for an answer no one will
    read. The Horde API cancels via DELETE /generate/text/status/{id}.
    Best-effort on purpose: a failed cancel must never mask the real
    error that abandoned the job, so every failure is swallowed.
    """
    try:
        client.delete(f"{api}/generate/text/status/{job_id}", headers=headers)
        logger.info("horde: cancelled orphaned job %s", job_id)
    except Exception:
        logger.debug("horde: cancel of job %s failed (best-effort)", job_id, exc_info=True)


async def _acancel_job(client, api: str, headers: dict, job_id: str) -> None:
    """Async twin of _cancel_job — same best-effort DELETE."""
    try:
        await client.delete(f"{api}/generate/text/status/{job_id}", headers=headers)
        logger.info("horde: cancelled orphaned job %s", job_id)
    except Exception:
        logger.debug("horde: cancel of job %s failed (best-effort)", job_id, exc_info=True)


async def _ahorde_generate_on(api: str, headers: dict, submit_body: dict, timeout_s: int) -> str:
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": HORDE_USER_AGENT}) as client:
        sub = await client.post(f"{api}/generate/text/async", headers=headers, json=submit_body)
        if sub.status_code == 429:
            # One bounded retry on the same host when the edge says how
            # long to wait: a short rate-limit blip should not burn a
            # host failover. No/invalid Retry-After (or a second 429)
            # fails over immediately via _check_submit_response.
            wait = _retry_after_s(sub)
            if wait is not None:
                logger.info("horde: rate limited on submit, retrying once after %.1fs", wait)
                await asyncio.sleep(wait)
                sub = await client.post(f"{api}/generate/text/async", headers=headers, json=submit_body)
        _check_submit_response(sub)
        job_id = sub.json().get("id")
        if not job_id:
            raise RuntimeError("horde: no job id returned")

        poll_headers = {"apikey": headers["apikey"], "User-Agent": HORDE_USER_AGENT}
        deadline = time.time() + timeout_s
        bad_polls = 0
        last_status: dict | None = None
        try:
            while time.time() < deadline:
                await asyncio.sleep(_poll_interval_s(last_status))
                chk = await client.get(f"{api}/generate/text/status/{job_id}", headers=poll_headers)
                if chk.status_code == 403:
                    # The job may still be queued — cancel it before raising
                    # so it does not keep occupying a worker.
                    await _acancel_job(client, api, poll_headers, job_id)
                    raise RuntimeError("horde: 403 forbidden on status poll")
                if chk.status_code != 200:
                    # Fail fast on a persistently sick status endpoint: a
                    # 429/5xx on every poll means this host is not serving
                    # jobs right now, and swallowing it burns the whole
                    # timeout (60s) before the router can fail over to
                    # Pollinations — while also misreporting the failure as
                    # a slow queue ("timed out"). Three bad polls in a row
                    # (~12s) is a real outage, not a blip: cancel the job
                    # and raise so the host failover tries the next Horde
                    # host. A single bad poll is tolerated (the counter
                    # resets on any good parse).
                    bad_polls += 1
                    if bad_polls >= POLL_ERROR_THRESHOLD:
                        await _acancel_job(client, api, poll_headers, job_id)
                        raise RuntimeError(
                            f"horde: status poll failed {bad_polls}x in a row (last: {chk.status_code})"
                        )
                    continue
                try:
                    s = chk.json()
                except ValueError:
                    # A 200 that is not JSON is transient garbage, not a
                    # dead host: count it like any other bad poll instead
                    # of aborting the whole call on one corrupt reply.
                    # (The counter resets only on a good parse below — a
                    # reset here would cap it at 1 and never fail fast.)
                    bad_polls += 1
                    if bad_polls >= POLL_ERROR_THRESHOLD:
                        await _acancel_job(client, api, poll_headers, job_id)
                        raise RuntimeError(
                            f"horde: status poll returned non-JSON {bad_polls}x in a row"
                        )
                    continue
                bad_polls = 0
                last_status = s
                _log_queue_progress(job_id, s)
                text = _parse_status_payload(s)
                if text is not None:
                    return text
        except asyncio.CancelledError:
            # The chat stream was cancelled (client disconnected, turn
            # timed out at the agent level): the job would otherwise
            # keep running on a worker for an answer nobody reads.
            # Shield the DELETE so the in-flight cancellation does not
            # cut the cleanup short, then re-raise so the cancellation
            # still propagates to the caller.
            await asyncio.shield(_acancel_job(client, api, poll_headers, job_id))
            raise
        # Deadline hit: the job is still queued/running on Horde — cancel
        # it instead of orphaning worker capacity on a job nobody waits for.
        await _acancel_job(client, api, poll_headers, job_id)
        # The models we queued on may be dead: drop the cached live list
        # so the next turn re-refreshes instead of re-queueing on them.
        invalidate_models_cache()
    raise RuntimeError("horde: timed out waiting for generation")


def _horde_generate_on(api: str, headers: dict, submit_body: dict, timeout_s: int) -> str:
    with httpx.Client(timeout=30, headers={"User-Agent": HORDE_USER_AGENT}) as client:
        sub = client.post(f"{api}/generate/text/async", headers=headers, json=submit_body)
        if sub.status_code == 429:
            # One bounded retry on the same host — see the async twin.
            wait = _retry_after_s(sub)
            if wait is not None:
                logger.info("horde: rate limited on submit, retrying once after %.1fs", wait)
                time.sleep(wait)
                sub = client.post(f"{api}/generate/text/async", headers=headers, json=submit_body)
        _check_submit_response(sub)
        job_id = sub.json().get("id")
        if not job_id:
            raise RuntimeError("horde: no job id returned")

        poll_headers = {"apikey": headers["apikey"], "User-Agent": HORDE_USER_AGENT}
        deadline = time.time() + timeout_s
        bad_polls = 0
        last_status: dict | None = None
        while time.time() < deadline:
            time.sleep(_poll_interval_s(last_status))
            chk = client.get(f"{api}/generate/text/status/{job_id}", headers=poll_headers)
            if chk.status_code == 403:
                # The job may still be queued — cancel it before raising
                # so it does not keep occupying a worker.
                _cancel_job(client, api, poll_headers, job_id)
                raise RuntimeError("horde: 403 forbidden on status poll")
            if chk.status_code != 200:
                # Fail fast, same reason as the async twin above: a
                # persistently sick status endpoint must trigger host
                # failover now, not after the whole timeout is burned.
                # Cancel the job first so the worker is freed.
                bad_polls += 1
                if bad_polls >= POLL_ERROR_THRESHOLD:
                    _cancel_job(client, api, poll_headers, job_id)
                    raise RuntimeError(
                        f"horde: status poll failed {bad_polls}x in a row (last: {chk.status_code})"
                    )
                continue
            try:
                s = chk.json()
            except ValueError:
                # A 200 that is not JSON is transient garbage, not a
                # dead host: count it like any other bad poll instead
                # of aborting the whole call on one corrupt reply.
                # (The counter resets only on a good parse below — a
                # reset here would cap it at 1 and never fail fast.)
                bad_polls += 1
                if bad_polls >= POLL_ERROR_THRESHOLD:
                    _cancel_job(client, api, poll_headers, job_id)
                    raise RuntimeError(
                        f"horde: status poll returned non-JSON {bad_polls}x in a row"
                    )
                continue
            bad_polls = 0
            last_status = s
            _log_queue_progress(job_id, s)
            text = _parse_status_payload(s)
            if text is not None:
                return text
        # Deadline hit: the job is still queued/running on Horde — cancel
        # it instead of orphaning worker capacity on a job nobody waits for.
        _cancel_job(client, api, poll_headers, job_id)
        # The models we queued on may be dead: drop the cached live list
        # so the next turn re-refreshes instead of re-queueing on them.
        invalidate_models_cache()
    raise RuntimeError("horde: timed out waiting for generation")


# Chat-speed cap (was 240s): a single queued Horde job must never hold a
# chat turn for 4 minutes. 60s is the cap because the router in app/main.py
# already fails over to the next provider on error — a Horde timeout is a
# failover trigger, not a dead end — and a queued anonymous job that has
# not started after 60s (anon queue measured 90+ deep) is unlikely to be
# the fastest path to an answer. The agent loop can call the model several
# times per turn (plan, tool calls, final), so 60s per call still allows
# multi-step tool use while keeping the worst case far under the 600s turn
# timeout; 240s per call could eat the whole turn budget in 3 calls.
# Override with HORDE_TIMEOUT_S (seconds) when a slower queue is preferred.
HORDE_CHAT_TIMEOUT_S = 60


def _default_horde_timeout() -> int:
    try:
        return max(1, int(os.environ.get("HORDE_TIMEOUT_S", HORDE_CHAT_TIMEOUT_S)))
    except ValueError:
        return HORDE_CHAT_TIMEOUT_S


class HordeChatModel(BaseChatModel):
    """LangChain chat model backed by AI Horde's keyless uncensored workers,
    with prompt-based tool-call emulation so UniHarness agents can use it."""

    horde_models: list[str] = HORDE_UNCENSORED_MODELS
    max_context_length: int = 8192
    max_length: int = 1500
    temperature: float = 0.3
    timeout_s: int = Field(default_factory=_default_horde_timeout)

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
        header, convo = _messages_to_parts(messages, tools_text)
        # Keep the prompt inside the worker's context window, trimming
        # oldest-first: the old code cut the END (prompt[:max_chars]),
        # which dropped the newest user turn — the message the model has
        # to answer — while keeping stale history.
        max_chars = self.max_context_length * 3  # ~3 chars per token, conservative
        return _fit_prompt(header, convo, max_chars)

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
            models=resolve_models(self.horde_models),
            max_context_length=self.max_context_length,
            max_length=self.max_length,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
            stop=stop,
        )
        return self._to_chat_result(text)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async twin of _generate — non-blocking Horde polling.

        LangChain calls this from async chains (like the agent's ainvoke);
        without it the sync _generate runs in a threadpool while still
        sleeping in 4s increments — this version yields to the event loop
        between polls so the API stays responsive mid-generation.
        """
        prompt = self._build_prompt(messages)
        logger.info("horde: submitting prompt async (%d chars)", len(prompt))
        text = await _ahorde_generate(
            prompt,
            models=resolve_models(self.horde_models),
            max_context_length=self.max_context_length,
            max_length=self.max_length,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
            stop=stop,
        )
        return self._to_chat_result(text)

    def _to_chat_result(self, text: str) -> ChatResult:
        clean, calls = _parse_tool_calls(text)
        tool_calls = [
            {"name": c["name"], "args": c["args"], "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
            for c in calls
        ]
        logger.info("horde: got %d chars, %d tool calls", len(text), len(tool_calls))
        message = AIMessage(content=clean, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])
