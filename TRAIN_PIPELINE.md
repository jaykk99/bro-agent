# bro-agent Training Pipeline

Fine-tune jobs queued **in bro-agent**, trained on **free GPU workers** (Google
Colab or Kaggle). Same pull-based system as the Forge Asset Pipeline, but fully
inside bro-agent: SQLite queue, FastAPI endpoints, adapters stored on bro-agent's
own disk. No Supabase, no Error Inbox, no Hugging Face, no third-party host.

## Architecture

```
bro-agent app (https://bro-agent.onrender.com)
  |  POST /api/train {model_repo, dataset, lora_rank, access_code}
  v
app/train.py  ->  SQLite train_jobs in data/bro.db (queued)
                                              |
                                              |  pull: POST /api/train/claim (every 30s)
                                              v
                              GPU worker: notebooks/train-worker.ipynb
                              (Google Colab free T4  OR  Kaggle GPU T4)
                                1. claim oldest queued job (atomic UPDATE..RETURNING)
                                2. Unsloth QLoRA fine-tune (r = job's lora_rank)
                                3. upload LoRA adapter -> POST /api/train/{id}/upload
                                   (stored at data/adapters/<job_id>/ on bro-agent)
                                4. POST /api/train/{id}/complete {status, result_url, metrics}
                                   — or failed (error)
                                5. WIPE local job dir + CUDA cache (disk is scratch)
                                              |
  GET /api/train/{id}                         |  result_url
  v                                           v
status, metrics, adapter download  <----  /api/train/{id}/files/adapter_model.safetensors
```

## Endpoints (all behind the access-code gate)

- `POST /api/train` — queue a job. Body: `{model_repo, dataset?, lora_rank? (1-256, default 16), access_code}` → `{job_id}`
- `GET /api/train/list?access_code=` — recent jobs (newest first, 50 max), each with `attempts` and `claimed_by` (holding worker id, or null)
- `GET /api/train/{id}?access_code=` — one job: status, result_url, metrics, error, attempts, claimed_by
- `POST /api/train/claim` — worker only. Body: `{access_code, worker_id?}` → oldest queued job, atomically marked `working`; 404 when the queue is empty. The claim records `worker_id` as the job's owner (`claimed_by`); heartbeat/complete then refuse any other worker with 409, so a stale worker can't report on a job another worker now holds. A claim without `worker_id` stays unowned (direct-claim users, old notebooks) and any reporter may complete it, as before.
- `POST /api/train/{id}/heartbeat` — worker only, while training. Body: `{access_code, worker_id?}` → `{ok}`; refreshes the job's `updated_at` so the reaper does not treat a legit long run as stuck. 404 when the job is missing or not `working` (a late heartbeat never revives a reaped/finished job); 409 when the job is held by another worker
- `POST /api/train/{id}/complete` — worker only. Body: `{access_code, status: "done"|"failed", result_url?, metrics?, error?, worker_id?}`. 404 when the job is missing or not `working`; 409 when the job is held by another worker
- `POST /api/train/{id}/upload` — worker only. Multipart form: `file` + `access_code`. Filenames are sanitized server-side (path traversal safe). Per-file cap 2 GB by default (413 + partial file removed over the cap); override with `BRO_ADAPTER_MAX_BYTES`
- `GET /api/train/{id}/files/{name}?access_code=` — download an adapter file
- `POST /api/train/{id}/retry` — re-queue a `failed` job (body: `{access_code}`).
  Attempts are tracked per job and capped at 3 — a poison job stops retrying
  instead of spinning forever. The same cap applies to the automatic paths:
  a job gets at most 3 re-queues across reap-stuck, the forge reaper, and
  /retry, and an exhausted job is failed instead of re-queued
- `POST /api/train/reap-stuck` — re-queue jobs stuck in `working` whose worker
  vanished (body: `{access_code}`), or fail them when they are out of
  attempts. Returns `{"requeued": [...], "failed": [...]}`. Covers direct
  `/api/train/claim` users the forge reaper can't see

## Worker reliability (forge dispatcher)

The worker notebook registers itself at
`POST /api/forge/workers/register` (platform auto-detected: colab/kaggle/other)
and sends a heartbeat every 30s carrying its current job id
(`POST /api/forge/workers/heartbeat`). When it finishes a job it reports
`POST /api/forge/workers/{id}/finished`.

If heartbeats stop for 10 minutes, `POST /api/forge/dispatch/reap` marks the
worker dead and re-queues its stuck job (`working` → `queued`, attempts+1) —
or fails the job if it already used its 3 attempts, so a poison job retires
instead of cycling through workers forever. A dead worker id never revives:
if the runtime wakes up after being reaped, its heartbeat is refused and it
must re-register for a fresh id (the notebook does this automatically) —
reviving it would restore its stale job id and the next reap would re-queue
the new holder's job. `POST /api/train/{id}/complete`
only accepts a report from the worker holding the job (`working`); a stale
worker reporting late can no longer overwrite another worker's result.
The ops dashboard in the UI shows alive/dead workers per platform, queue
depth, and has reap buttons — or call the endpoints directly.

## Auto-reaper (background loop)

Reaping used to be manual-only, so a worker that died without notice left
its job `working` until someone remembered to call reap. The app now runs
both reaps itself on a timer (`app/reaper.py`, started from the FastAPI
lifespan): one pass on boot, then every `BRO_REAP_INTERVAL_S` (default
600s; 0 disables). A pass re-queues `working` jobs whose worker vanished
and marks dead forge workers dead, re-queuing their stuck jobs; a job at
the attempts cap (3) is failed instead of re-queued, so a poison job
retires instead of cycling. `GET /api/forge/workers` reports the reaper
state (`enabled`, `interval_s`, `running`, `runs`, `last_run_at`,
`last_result`, `last_error`). The manual endpoints still work unchanged
and are safe to call alongside the loop — reaping is idempotent, so the
second caller just finds nothing to do.

## Why this shape

- **Runs in the app, trains on Colab/Kaggle.** Queue + status + storage live in
  bro-agent; heavy compute happens on the free GPU worker. No GPU bill, no
  extra server.
- **Pull-based worker.** Colab/Kaggle runtimes are ephemeral (timeouts, idle
  disconnects). The worker pulls, so a restart loses nothing — jobs stay
  `queued`/`working` in bro-agent's SQLite and get re-claimed. The atomic
  `UPDATE..RETURNING` claim means two workers never double-process a job.
- **Own infra only.** Adapters land in `data/adapters/<job_id>/` on bro-agent's
  own disk (Render free tier: ephemeral — see limits). The worker disk is
  scratch: every job's files are deleted right after upload (the user's rule,
  enforced in the loop's finally block).
- **Guards untouched.** `app/guards.py` is not modified or bypassed by this
  system; training endpoints don't take chat text. The access-code gate is the
  auth layer.

## Data flow per job

1. Queue: `POST /api/train {"model_repo":"Qwen/Qwen2-1.5B","dataset":"user/my-data","lora_rank":16,"access_code":"..."}` → `{job_id}`.
2. Worker claims it (atomic), trains (~30–120 min on a T4 for 1–8B models):
   Unsloth 4-bit QLoRA, `use_rslora=True`, cosine schedule, 60 steps default
   (`TS_MAX_STEPS` env). While training it heartbeats the job every 30s
   (`POST /api/train/{id}/heartbeat`) so the 60-min stuck-job reaper does
   not re-queue it mid-run. Dataset auto-formats: Alpaca (`instruction`/`input`/
   `response`) → chat (`messages`) → raw text (first column).
3. Adapter files (`adapter_model.safetensors`, config, tokenizer) upload to
   bro-agent; job marked `done` with `result_url` + `metrics_json`
   (`{train_loss, examples, max_steps, lora_rank}`), or `failed` with `error`.
4. Worker wipes its local job dir and CUDA cache.
5. Poll `GET /api/train/{id}` for status; download the adapter from
   `result_url`.

## Setup

1. Deploy bro-agent with the new code (Render redeploys on push). Needs
   `python-multipart` (added to requirements.txt). `BRO_ACCESS_CODE` must be
   set (it already is).
2. Worker: open `notebooks/train-worker.ipynb` in Colab (T4) or Kaggle (GPU
   T4, **Internet ON**), add secrets `BRO_BASE_URL` + `BRO_ACCESS_CODE`, Run
   all. Optional env: `TS_MAX_STEPS`, `TS_POLL_SECS`.
3. Queue jobs via `POST /api/train`. Run one worker per GPU source for
   parallel training.

## Cost

$0. Colab free tier, Kaggle free tier, bro-agent's existing hosting. No paid
services.

## Limits (honest)

- **Stuck-job reaping (60 min rule).** A `working` job untouched for 60 min
  is re-queued (attempts+1) by the auto-reaper, or failed at the attempts
  cap (3) — so a dead worker's job never blocks the queue. But a training
  run takes 30–120 min, so the worker MUST heartbeat the job itself
  (`POST /api/train/{id}/heartbeat`, every 30s from the notebook's heartbeat
  loop) while training; without it a legit long run is reaped mid-training
  and a second worker trains it again. The forge worker heartbeat alone
  does not count — it keeps the worker row alive, not the train job.
- **Render free-tier disk is ephemeral** — adapters in `data/adapters/` vanish
  on redeploy/restart. Download finished adapters promptly, or point
  `BRO_DB_PATH`/adapters at a persistent disk later. Same for the SQLite DB
  itself (pre-existing bro-agent caveat, not new).
- **Kaggle GPU needs a phone-verified account; ~30 GPU-hours/week, 2
  concurrent sessions max.** Colab free T4s time out (~12h) and drop when
  idle. Throughput ≈ one 1–8B LoRA job per 1–2h per worker.
- **Render sleeps when idle** — the first worker poll after idle can take
  ~60s to wake the app; the worker retries automatically.
- Unsloth weights + datasets download on first run (~10–20 GB); the wipe
  rule covers only *job* files, not the model/pip cache.
- One job at a time per notebook; scale by running more notebooks.
- Dataset auto-format is best-effort; clean Alpaca/chat datasets work best.
