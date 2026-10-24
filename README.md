# bro-agent

Jay's personal uncensored agent app. Standalone — its own repo, UI, and deployment.

Built on [UniHarness](https://github.com/UnicomAI/UniHarness) (MIT): the agent gets a real
computer (terminal + filesystem), web browsing, and Composio integrations.

## Models (uncensored only — never GPT/Claude/Gemini)

1. **Pollinations** (primary) — keyless, OpenAI-compatible, native tool
   calling. No key needed.
2. **AI Horde** (fallback) — keyless, genuinely uncensored community workers
   (Stheno / Heretic / Rocinante lineage). Tool calling is emulated
   prompt-side since Horde workers don't do native function calling.
   Slower than Pollinations, but uncensored with zero keys.
   No Venice dependency. If a model refuses, the request is skipped to the
   next model; a refusal is never shown to the user as the answer.

## Setup

```bash
pip install -r requirements.txt
export COMPOSIO_API_KEY=...      # optional, for Composio tools
export BRO_ACCESS_CODE=...       # optional access gate for the UI
uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

## Deploy (Render)

`render.yaml` is included. Set `VENICE_API_KEY`, `COMPOSIO_API_KEY`, and
`BRO_ACCESS_CODE` in the Render dashboard (never commit keys).

## Guardrails

7 surgical guards (ported from error-inbox PR #209) block only unambiguous
hard-boundary intent: stolen cards, phishing/fraud, wallet theft, unauthorized
hacking, weapons of mass harm, CSAM, doxxing. Each needs a criminal action +
criminal target — bare keywords never match, legit work always passes.
Everything else: zero refusal boilerplate.

## Training jobs and Forge workers

Pull-based GPU jobs, all behind the access-code gate:

- `POST /api/train` — queue a fine-tune job (`model_repo`, `dataset`,
  `lora_rank`). `GET /api/train/list`, `GET /api/train/{id}`.
- `POST /api/train/claim` — a worker claims the oldest queued job
  (atomic, first claimer wins). `POST /api/train/{id}/complete` and
  `POST /api/train/{id}/upload` report results and adapters back.
- `POST /api/forge/workers/register` — a Colab/Kaggle worker registers.
- `POST /api/forge/workers/heartbeat` — worker heartbeat with its current
  job, if any. `POST /api/forge/workers/{id}/finished` clears it.
- `GET /api/forge/workers` — dispatcher status: workers alive per
  platform plus queue depth.
- `POST /api/forge/dispatch/reap` — marks workers dead after 180s without
  a heartbeat and re-queues their stuck Supabase jobs (only when
  `SUPABASE_URL` plus `SUPABASE_SERVICE_KEY`/`SUPABASE_KEY` are set;
  otherwise it reports re-queueing as skipped, honestly).

Full worker flow: see `TRAIN_PIPELINE.md`.

## Memory

SQLite at `./data/bro.db` (or `BRO_DB_PATH`). Threads persist across restarts
on persistent disks; Render free-tier disk is ephemeral.
