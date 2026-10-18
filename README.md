# bro-agent

Jay's personal uncensored agent app. Standalone — its own repo, UI, and deployment.

Built on [UniHarness](https://github.com/UnicomAI/UniHarness) (MIT): the agent gets a real
computer (terminal + filesystem), web browsing, and Composio integrations.

## Models (uncensored only — never GPT/Claude/Gemini)

1. **Pollinations** (primary): keyless, OpenAI-compatible, native tool
   calling. Open weights, no refusals on legit work. No API key needed.
2. **AI Horde** (fallback): keyless, genuinely uncensored community workers
   (Stheno / Heretic / Rocinante lineage). Tool calling is emulated
   prompt-side since Horde workers don't do native function calling.
   Slower than Pollinations, but uncensored with zero keys. A free
   `HORDE_API_KEY` (optional) gets better queue priority.

## Setup

```bash
pip install -r requirements.txt
export COMPOSIO_API_KEY=...      # optional
export BRO_ACCESS_CODE=...       # optional access gate for the UI
export HORDE_API_KEY=...         # optional, better Horde queue priority
uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

## Deploy (Render)

`render.yaml` is included. Set `COMPOSIO_API_KEY` and `BRO_ACCESS_CODE` in
the Render dashboard (`HORDE_API_KEY` optional). Never commit keys.

## Guardrails

7 surgical guards (ported from error-inbox PR #209) block only unambiguous
hard-boundary intent: stolen cards, phishing/fraud, wallet theft, unauthorized
hacking, weapons of mass harm, CSAM, doxxing. Each needs a criminal action +
criminal target — bare keywords never match, legit work always passes.
Everything else: zero refusal boilerplate.

## Memory

SQLite at `./data/bro.db` (or `BRO_DB_PATH`). Threads persist across restarts
on persistent disks; Render free-tier disk is ephemeral.

## Live URL

https://bro-agent.onrender.com
