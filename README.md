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

The router (`app/router.py`) orders providers healthy-first: two consecutive
failures cools a provider down for 5 minutes. If a fallback model's turn comes
back as refusal boilerplate on work the hard-boundary pre-check did NOT block,
the turn is retried on the next provider — Jay never sees a "sorry, I can't"
on legit work (his standing order). The Horde model list refreshes against the
live Horde API every 6h so jobs don't queue forever on dead models.

## Setup

```bash
pip install -r requirements.txt
export COMPOSIO_API_KEY=<redacted>
export BRO_ACCESS_CODE=...       # optional access gate for the UI
export HORDE_API_KEY=<redacted>
export BRO_WORKDIR=~/bro-work    # agent sandbox (default)
uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

## Deploy (Render)

`render.yaml` is included. Set `COMPOSIO_API_KEY` and `BRO_ACCESS_CODE` in
the Render dashboard (`HORDE_API_KEY` optional). Never commit keys.

## What the agent can do (21 tools)

**Web** — `web_fetch` (SSRF-hardened page fetch), `web_search` (DuckDuckGo, keyless).
Fetched content is wrapped as untrusted and any planted `tool_call` fences are
neutralized, so hostile pages can't drive the agent's tools.

**Files** (all jailed to `BRO_WORKDIR`) — `file_read`, `file_write`,
`file_edit` (exact single-anchor replace), `file_list`, `file_search` (regex
grep), `file_mkdir`.

**Shell** (jailed, timed out, catastrophic patterns blocked) — `run_shell`,
`run_python`, plus `start_bg` / `bg_poll` / `bg_kill` for long-running jobs.

**Git** (repos under the workdir) — `git_status`, `git_diff`, `git_log`,
`git_add`, `git_commit`, `git_branch`, `git_push`, `git_pull` (`--ff-only`).
Force-push, hard reset, `clean -f`, and history rewrites are refused in code.

**Composio** — `composio_execute`, `composio_apps`, `composio_list_tools`
(discover tool slugs per app instead of guessing).

## Chat UI

Markdown rendering (code blocks with copy buttons), thread sidebar with
rename/delete, timestamps, per-message copy, typing indicator naming the
active provider, error retry, rate-limit notice, and an **ops dashboard**:
train-job queue (retry/reap), forge worker status, and usage stats.
Access code is remembered in the browser.

## Guardrails

7 surgical guards (ported from error-inbox PR #209, living in
`app/refusal.py` and only there) block only unambiguous hard-boundary intent:
stolen cards, phishing/fraud, wallet theft, unauthorized hacking, weapons of
mass harm, CSAM, doxxing. Each needs a criminal action + criminal target —
bare keywords never match, legit work always passes. Everything else: zero
refusal boilerplate. See `docs/SECURITY.md` for the full security model.

## API

Chat: `POST /api/chat` (SSE stream, 30 turns/min rate-limited).
Threads: list / messages / rename / delete / Markdown export.
Models: `GET /api/models` (chain health), `GET /api/stats` (usage counts).
Uploads: `POST /api/upload` (into `workdir/uploads/`).
Training + forge worker endpoints: see the module docstring in `app/main.py`.

## Training pipeline

Pull-based GPU fine-tuning lives inside bro-agent (no third parties):
`POST /api/train` queues a job, a Colab/Kaggle worker
(`notebooks/train-worker.ipynb`) claims it over HTTP, trains with Unsloth
QLoRA, uploads the LoRA adapter back, wipes its disk. Failed jobs can be
retried (`/api/train/{id}/retry`, capped at 3 attempts); jobs stuck
`working` with a vanished worker are re-queued (`/api/train/reap-stuck`).
See `TRAIN_PIPELINE.md`.

## Memory

SQLite at `./data/bro.db` (or `BRO_DB_PATH`). Threads persist across restarts
on persistent disks; Render free-tier disk is ephemeral.

## Tests

```bash
pip install pytest httpx
python -m pytest tests/ -q --ignore=tests/test_horde.py
```

`tests/test_horde.py` needs `langchain_core` installed (production dep);
everything else runs dependency-free thanks to the `app/tool_base.py` shim.

## Live URL

https://bro-agent.onrender.com
