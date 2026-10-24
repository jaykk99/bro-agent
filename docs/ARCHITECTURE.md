# bro-agent architecture

Single-process FastAPI app (`app/main.py`) + UniHarness agent
(`app/agent.py`) + SQLite (`app/memory.py`). No workers, no queues, no
external DB — Render free tier friendly.

```
browser UI (static/index.html)
        │  SSE stream / JSON
        ▼
app/main.py ──┬── access-code gate, rate limit, hard-boundary pre-check
              ├── context windowing (app/context.py)
              ├── model router (app/router.py): pollinations → horde
              └── agent turn: UniHarness create_agent(...)
                                   │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              app/tools_web.py  app/tools_files.py  app/tools_exec.py
              app/tools_git.py  app/tools_composio.py
                    │                │
                    ▼                ▼
              app/ssrf.py       app/sandbox.py   (shared guards)
```

## Request flow (`POST /api/chat`)

1. Access-code check → rate-limit check (429 with `Retry-After`).
2. Empty-message 400.
3. Hard-boundary pre-check (`app/refusal.py` via `app/guards.py`). A hit
   is stored and returned immediately — no model is ever consulted.
4. Thread resolved/created; user message stored; history loaded and
   windowed to a 24k-char budget (`build_window`).
5. The router's healthy-first provider order is walked:
   - build the agent pinned to that provider (`create_bro_agent(preferred=…)`)
   - `ainvoke` the turn
   - on error **or** refusal boilerplate from a fallback model → record
     the failure, try the next provider
   - on success → record success, store the reply, record usage stats
6. SSE events: `thread` (id), then `done` (reply + model) or `error`.

## Models

- **Pollinations** (`openai` model id → gpt-oss-20b): keyless,
  OpenAI-compatible, native tool calling. Built via `langchain_openai`.
- **AI Horde** (`app/horde_llm.py`): `HordeChatModel`, a LangChain chat
  model over the Horde text API. Workers don't do function calling, so
  `bind_tools` injects tool schemas as text and `_parse_tool_calls`
  extracts ` ```tool_call ` JSON fences back into `AIMessage.tool_calls`.
  Both sync `_generate` and async `_agenerate` exist (the event loop must
  never block on 4s poll sleeps); `_ahorde_generate_on` fails over across
  both Horde API hosts. `resolve_models()` refreshes the uncensored model
  list from the live Horde API every 6h.

## Storage (one SQLite file)

| module | tables | holds |
|---|---|---|
| `memory.py` | `threads`, `messages` | chat threads + turns |
| `train.py` | `train_jobs` | fine-tune job queue (queued/working/done/failed, attempts) |
| `dispatch.py` | `forge_workers` | Colab/Kaggle worker registry + heartbeats |
| `stats.py` | `chat_stats` | per-day/per-provider turn counts |

`BRO_DB_PATH` overrides the location (default `./data/bro.db`).

## Training pipeline (pull-based, inside this repo)

`POST /api/train` → row in `train_jobs` (queued). A GPU worker
(`notebooks/train-worker.ipynb`, Colab/Kaggle) polls `POST /api/train/claim`
(atomic claim — first claimer wins), trains with Unsloth QLoRA, uploads the
adapter (`POST /api/train/{id}/upload`), reports
(`POST /api/train/{id}/complete`), wipes its disk.

Reliability: the forge dispatcher (`app/dispatch.py`) registers workers and
reaps dead ones (10-min heartbeat timeout), re-queueing their stuck jobs;
`fail_stuck_jobs()` covers jobs stuck `working` with no worker row;
`retry_job()` re-queues failed jobs up to 3 attempts.

## Adding a tool

1. Create the tool class in the right `app/tools_*.py` (or a new one),
   subclassing `BaseAgentTool` from `app/tool_base.py` (falls back to a
   shim when UniHarness isn't installed, so tests stay runnable).
2. Keep tool logic importable without UniHarness — put pure logic in
   helper functions/modules.
3. Register it in the `*_tools()` factory and in `create_bro_agent()`.
4. Jail any filesystem/shell/network access via `app/sandbox.py` /
   `app/ssrf.py` — never trust agent-supplied paths or URLs directly.
5. Add tests under `tests/` (no network: monkeypatch DNS/HTTP).

## Test layout

`tests/test_*.py`, plain pytest. Network-dependent modules are tested with
`httpx.MockTransport` / monkeypatched resolvers; `test_horde_async.py`
injects minimal `langchain_core` stubs because that dep isn't installable
in every environment. `tests/test_horde.py` (older) needs the real
`langchain_core` and is excluded from the dependency-free run.
