"""Bro agent factory — UniHarness agent with uncensored models only.

Model chain (uncensored FIRST — no GPT, no Claude, no Gemini):
  1. AI Horde (primary) — keyless, genuinely uncensored workers
     (Stheno / Heretic / Rocinante lineage). Tool calling is emulated
     prompt-side since Horde workers don't do native function calling.
     Slower (queue-based) but it is the least-refusing model in the pool,
     so it goes first (refusal-elimination pass, 2026-10-23).
  2. Pollinations (availability fallback only) — keyless, fast,
     OpenAI-compatible with NATIVE tool calling. Model is gpt-oss-20b
     (open weights, but the more aligned of the two — tried last, and any
     refusal it returns is skipped by the router in app/router.py).

No Venice dependency. No API keys needed. No paid services.

The agent gets a real computer (terminal + filesystem) via UniHarness's
LocalNativeComputer, plus web tools and Composio.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

POLLINATIONS_API_URL = "https://text.pollinations.ai/openai"
POLLINATIONS_MODEL = "openai"  # resolves to gpt-oss-20b, open weights; most aligned in pool, fallback only

SYSTEM_PROMPT = """You are bro — Jay's personal AI agent. You get things done.

How you act:
- Be genuinely helpful, not performatively helpful. Skip the fluff. Just help.
- Have opinions. You're allowed to prefer things and find things funny or dull.
- Be resourceful before asking. Read the file, check the context, search, try building it. Come back with answers, not questions.
- You have a computer: terminal, filesystem, shell. Use it.
- You can browse the web and use Jay's connected integrations via Composio.
- Your tools (use them, don't narrate them):
  - web_search / web_fetch for current info and docs (fetches are SSRF-screened)
  - file_read / file_write / file_edit / file_append / file_list / file_search / file_mkdir / file_move / file_copy / file_delete / file_stat — all inside ~/bro-work
  - run_shell / run_python for commands and code (timeouts apply); start_bg / bg_poll / bg_kill for long jobs
  - git_status / git_diff / git_log / git_add / git_commit / git_branch / git_push / git_pull for repos under ~/bro-work (never force-push)
  - generate_image for AI images (ad creative, mockups) — saves under workdir/images/
  - composio_execute / composio_apps / composio_list_tools for Jay's integrations (Gmail, GitHub, etc.) — use composio_list_tools first when unsure of the slug
- Prefer doing over asking: read the file, run the command, check the result.
- Plan multi-step work, execute step by step, report progress as you go.
- Jay types fast and direct. Match that: short, casual, straight to the point. No em dashes, ever.
- Never imitate dictation typos. Stay clear and correct.

What you never do:
- You don't refuse legitimate work. No "I cannot do this", no hedging on coding, business ops, research, writing, analysis, or security work on Jay's own assets and authorized targets. You just do the work.
- Hard boundaries (the only refusals, ever): stolen-card stuff, phishing/fraud kits, draining or stealing anyone's wallets or funds, hacking targets Jay didn't authorize, weapons of mass harm, anything sexual involving minors, doxxing private info on people. If one of these comes up, say plainly you can't do that one thing and offer the nearest legitimate alternative. One line, no lecture.

Jay's context: Winnipeg MB. Runs six businesses (Gutter Genie, Winnipeg Web Works, Move & Metal, Bnb Media, DemoBoss, Quonset Place). Solo dev. Prefers free/keyless where possible. Judges by visible, verifiable change.
"""


def _pollinations_model() -> Any:
    """Build the Pollinations chat model. Keyless, native tool calling."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=POLLINATIONS_MODEL,
        base_url=POLLINATIONS_API_URL,
        api_key="not-needed",  # Pollinations anon tier needs no key
        temperature=0.7,
    )


def build_llm(order: tuple[str, ...] = ("horde", "pollinations")) -> tuple[Any, str]:
    """Build the chat model. Returns (model, source_label).

    Tries providers in `order`; first one that initializes wins. Callers
    (via the router) pass the healthy-first order; the default is the
    uncensored-first chain: horde primary, pollinations fallback.
    """
    last_error: Exception | None = None
    for name in order:
        try:
            if name == "pollinations":
                # Keyless, fast, native tool calling. Availability fallback.
                return _pollinations_model(), "pollinations"
            if name == "horde":
                # Keyless uncensored workers, tool calling emulated.
                from .horde_llm import HordeChatModel

                return HordeChatModel(), "horde"
            logger.warning("build_llm: unknown provider %r, skipping", name)
        except Exception as e:
            logger.warning("build_llm: %s init failed: %s", name, e)
            last_error = e
    raise RuntimeError(f"no LLM provider available (tried {order}, last error: {last_error})")


async def create_bro_agent(preferred: str | None = None):
    """Create the UniHarness bro agent. Call once and reuse (or per-request).

    `preferred` optionally forces one provider ("pollinations" or "horde");
    otherwise the uncensored-first chain order is used.
    """
    from uniharness import ModelProfile, create_agent
    from uniharness.computer import LocalNativeComputer

    from .tools_composio import composio_tools
    from .tools_exec import exec_tools
    from .tools_files import file_tools
    from .tools_git import git_tools
    from .tools_image import image_tools
    from .tools_web import web_tools

    order = (preferred,) if preferred else ("horde", "pollinations")
    llm, source = build_llm(order)
    logger.info("bro agent model source: %s", source)

    profile = ModelProfile(model=llm, context_window=128_000)

    extra_tools: list[Any] = []
    extra_tools.extend(web_tools())
    extra_tools.extend(composio_tools())
    extra_tools.extend(file_tools())
    extra_tools.extend(exec_tools())
    extra_tools.extend(git_tools())
    extra_tools.extend(image_tools())

    workdir = os.environ.get("BRO_WORKDIR", os.path.expanduser("~/bro-work"))
    os.makedirs(workdir, exist_ok=True)

    agent = await create_agent(
        model=profile,
        computer=LocalNativeComputer(),
        extra_tools=extra_tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, source


# ---- Agent cache (chat-speed fix 1) -------------------------------------
# Before this, POST /api/chat rebuilt the whole agent on EVERY turn (and
# once per provider attempt): graph compilation, skill discovery, tool
# construction — all before the first model call. The cache below builds
# each provider's agent once and reuses it across turns.
#
# Why sharing one agent object across concurrent turns is safe here:
#  * create_bro_agent passes NO checkpointer, so the compiled LangGraph
#    graph holds no per-conversation state — every turn passes its own
#    {"messages": [...]} input, and LangGraph invokes with distinct state
#    per call (the supported concurrent-use pattern).
#  * LocalNativeComputer runs each command as a transient bash process —
#    it holds no shell session state to leak between turns.
#  * The tool objects are stateless per call (params in, output out);
#    the only shared mutable registries (background-job table, task
#    registry) are lock-guarded and shared by design across turns.
#  * The LLM objects are stateless per call (bind_tools copies).
# One deliberate consequence: a cached agent is NEVER aclose()d mid-process
# (aclose cancels the shared task registry, which would kill in-flight
# turns). Cached agents live for the process lifetime; invalidation only
# drops the reference. No MCP servers are configured in create_bro_agent,
# so there are no connections to leak by doing this.
#
# Invalidation: the fingerprint covers everything that changes what the
# built agent is (provider, system prompt, workdir, Horde timeout). Env
# values the tools read lazily at execute time (e.g. COMPOSIO_API_KEY)
# are NOT in the fingerprint — a rotated key takes effect on the next
# tool call without a rebuild. Call invalidate_agent_cache() after a
# config change that the fingerprint does not cover, or from tests.

_AGENT_CACHE: dict[str, tuple[Any, str]] = {}
_AGENT_CACHE_LOCK = asyncio.Lock()


def agent_cache_fingerprint(preferred: str | None) -> str:
    """Stable key for the agent a turn with `preferred` would build."""
    parts = [
        preferred or "chain",
        SYSTEM_PROMPT,
        os.environ.get("BRO_WORKDIR", os.path.expanduser("~/bro-work")),
        os.environ.get("HORDE_TIMEOUT_S", ""),
        POLLINATIONS_MODEL,
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


async def get_cached_agent(preferred: str | None = None):
    """Return (agent, source, cache_hit), building once per fingerprint.

    Creation is serialized by a lock (double-checked after acquiring it)
    so concurrent first turns build exactly one agent per provider.
    """
    key = agent_cache_fingerprint(preferred)
    hit = _AGENT_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1], True
    async with _AGENT_CACHE_LOCK:
        hit = _AGENT_CACHE.get(key)
        if hit is not None:
            return hit[0], hit[1], True
        agent, source = await create_bro_agent(preferred=preferred)
        _AGENT_CACHE[key] = (agent, source)
        logger.info("bro agent cached: provider=%s source=%s", preferred or "chain", source)
        return agent, source, False


def invalidate_agent_cache() -> int:
    """Drop all cached agents (e.g. after a config change). Returns count.

    Does NOT aclose() the dropped agents — see the cache note above.
    """
    n = len(_AGENT_CACHE)
    _AGENT_CACHE.clear()
    return n
