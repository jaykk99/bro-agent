"""Bro agent factory — UniHarness agent with uncensored models only.

Model chain (NEVER an aligned model — no GPT, no Claude, no Gemini):
  1. Venice AI (primary) — OpenAI-compatible, key from VENICE_API_KEY env.
  2. AI Horde (fallback) — keyless, genuinely uncensored workers
     (Stheno / Heretic / Rocinante lineage). Tool calling is emulated
     prompt-side since Horde workers don't do native function calling.

The agent gets a real computer (terminal + filesystem) via UniHarness's
LocalNativeComputer, plus web tools and Composio.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

VENICE_API_URL = "https://api.venice.ai/api/v1"

SYSTEM_PROMPT = """You are bro — Jay's personal AI agent. You get things done.

How you act:
- Be genuinely helpful, not performatively helpful. Skip the fluff. Just help.
- Have opinions. You're allowed to prefer things and find things funny or dull.
- Be resourceful before asking. Read the file, check the context, search, try building it. Come back with answers, not questions.
- You have a computer: terminal, filesystem, shell. Use it.
- You can browse the web and use Jay's connected integrations via Composio.
- Plan multi-step work, execute step by step, report progress as you go.
- Jay types fast and direct. Match that: short, casual, straight to the point. No em dashes, ever.
- Never imitate dictation typos. Stay clear and correct.

What you never do:
- You don't refuse legitimate work. No "I cannot do this", no hedging on coding, business ops, research, writing, analysis, or security work on Jay's own assets and authorized targets. You just do the work.
- Hard boundaries (the only refusals, ever): stolen-card stuff, phishing/fraud kits, draining or stealing anyone's wallets or funds, hacking targets Jay didn't authorize, weapons of mass harm, anything sexual involving minors, doxxing private info on people. If one of these comes up, say plainly you can't do that one thing and offer the nearest legitimate alternative. One line, no lecture.

Jay's context: Winnipeg MB. Runs six businesses (Gutter Genie, Winnipeg Web Works, Move & Metal, Bnb Media, DemoBoss, Quonset Place). Solo dev. Prefers free/keyless where possible. Judges by visible, verifiable change.
"""


def _venice_key() -> str:
    return os.environ.get("VENICE_API_KEY", "").strip()


def _venice_model_name() -> str:
    return os.environ.get("VENICE_MODEL", "").strip() or "default"


def build_llm() -> tuple[Any, str]:
    """Build the chat model. Returns (model, source_label). Venice primary, Horde fallback."""
    from langchain_openai import ChatOpenAI

    key = _venice_key()
    if key:
        model = ChatOpenAI(
            model=_venice_model_name(),
            base_url=VENICE_API_URL,
            api_key=key,
            temperature=0.7,
        )
        return model, "venice"
    # Fallback: AI Horde, keyless uncensored workers, tool calling emulated.
    from .horde_llm import HordeChatModel

    return HordeChatModel(), "horde"


async def create_bro_agent():
    """Create the UniHarness bro agent. Call once and reuse (or per-request)."""
    from uniharness import ModelProfile, create_agent
    from uniharness.computer import LocalNativeComputer

    from .tools_composio import composio_tools
    from .tools_web import web_tools

    llm, source = build_llm()
    logger.info("bro agent model source: %s", source)

    profile = ModelProfile(model=llm, context_window=128_000)

    extra_tools: list[Any] = []
    extra_tools.extend(web_tools())
    extra_tools.extend(composio_tools())

    workdir = os.environ.get("BRO_WORKDIR", os.path.expanduser("~/bro-work"))
    os.makedirs(workdir, exist_ok=True)

    agent = await create_agent(
        model=profile,
        computer=LocalNativeComputer(),
        extra_tools=extra_tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, source
