"""Bro agent factory — UniHarness agent with uncensored models only.

Model chain (NEVER an aligned model — no GPT, no Claude, no Gemini):
  1. Pollinations (primary) — keyless, fast, OpenAI-compatible with NATIVE
     tool calling. Model is gpt-oss-20b (open weights, no refusals on legit work).
  2. AI Horde (fallback) — keyless, genuinely uncensored workers
     (Stheno / Heretic / Rocinante lineage). Tool calling is emulated
     prompt-side since Horde workers don't do native function calling.
     Slower (queue-based) but works when Pollinations is down.

No Venice dependency. No API keys needed. No paid services.

The agent gets a real computer (terminal + filesystem) via UniHarness's
LocalNativeComputer, plus web tools and Composio.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

POLLINATIONS_API_URL = "https://text.pollinations.ai/openai"
POLLINATIONS_MODEL = "openai"  # resolves to gpt-oss-20b, open weights, uncensored

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


def build_llm(order: tuple[str, ...] = ("pollinations", "horde")) -> tuple[Any, str]:
    """Build the chat model. Returns (model, source_label).

    Tries providers in `order`; first one that initializes wins. Callers
    (via the router) pass the healthy-first order; the default keeps the
    classic pollinations-primary, horde-fallback chain.
    """
    last_error: Exception | None = None
    for name in order:
        try:
            if name == "pollinations":
                # Keyless, fast, uncensored, native tool calling.
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
    otherwise the classic chain order is used.
    """
    from uniharness import ModelProfile, create_agent
    from uniharness.computer import LocalNativeComputer

    from .tools_composio import composio_tools
    from .tools_exec import exec_tools
    from .tools_files import file_tools
    from .tools_git import git_tools
    from .tools_image import image_tools
    from .tools_web import web_tools

    order = (preferred,) if preferred else ("pollinations", "horde")
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
