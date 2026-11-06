"""bro-agent — Jay's personal uncensored agent app. FastAPI + UniHarness.

Endpoints:
  GET  /            — chat UI
  GET  /api/health  — liveness + model source + provider status
  GET  /api/models  — provider chain status (order, health, cooldowns)
  GET  /api/stats   — per-day/per-provider usage counts (access_code query param)
  GET  /api/threads — list threads (optional ?search= to match titles/messages)
  GET  /api/threads/{id} — thread messages with timestamps (?limit=&offset= for paging)
  POST /api/threads/{id}/rename — rename a thread. Body: {title, access_code}
  DELETE /api/threads/{id} — delete a thread and its messages
  GET  /api/threads/{id}/export — full thread as a Markdown download (paged, no 1000-message cap)
  POST /api/chat    — chat turn (SSE stream). Body: {message, thread_id?, access_code}
                     Events: thread, progress{stage: trying|failed|refused, provider},
                     token{delta} (live model tokens; Horde turns emit at most
                     one, at job completion — see app/chat_stream.py),
                     ping (20s heartbeat), done{reply, model}, error
  POST /api/upload  — upload a file into workdir/uploads/. Form: {file, access_code}
  GET  /api/uploads — list uploaded files (name, bytes, mtime)
  DELETE /api/uploads/{name} — delete one uploaded file
  POST /api/train   — queue a fine-tune job. Body: {model_repo, dataset?, lora_rank?, access_code}
  GET  /api/train/list — recent training jobs (access_code query param)
  GET  /api/train/{id} — one training job (access_code query param)
  POST /api/train/claim — worker claims oldest queued job. Body: {access_code}
  POST /api/train/{id}/complete — worker reports done/failed. Body: {access_code, status, ...}
  POST /api/train/{id}/retry — re-queue a failed job (attempts+1, capped). Body: {access_code}
  POST /api/train/reap-stuck — re-queue 'working' jobs whose worker vanished; jobs at the attempts cap fail instead (body: {access_code})
  POST /api/train/{id}/upload — worker uploads adapter file (multipart). Form: {file, access_code}
  GET  /api/train/{id}/files/{name} — download an adapter file (access_code query param)
  POST /api/forge/workers/register — worker registers (platform colab/kaggle). Body: {platform, access_code}
  POST /api/forge/workers/heartbeat — worker heartbeat. Body: {worker_id, current_job_id?, access_code}
  POST /api/forge/workers/{id}/finished — worker reports job done (clears current job). Body: {access_code}
  GET  /api/forge/workers — dispatcher status: workers alive per platform + queue depth + auto-reaper state (access_code query param)
  POST /api/forge/dispatch/reap — reap dead workers, re-queue their stuck jobs. Body: {access_code}
  (Auto-reaper: a background loop also runs both reaps every BRO_REAP_INTERVAL_S, default 600s; 0 disables. See app/reaper.py.)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import memory
from . import reaper
from . import train as train_jobs
from . import dispatch as forge_dispatch
from .context import build_window
from .guards import check_hard_boundary
from .guard_policy import check_owner_boundary
from . import rate_limit
from . import stats
from .router import get_router, looks_like_refusal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the auto-reaper on boot, stop it on shutdown.

    The reaper loop runs both stuck-job reaps on a timer (see
    app/reaper.py), so a dead worker's job gets re-queued without anyone
    remembering to POST the manual reap endpoints. Disabled with
    BRO_REAP_INTERVAL_S=0. The loop's first pass runs immediately, so a
    restart clears the backlog instead of waiting out the interval."""
    reaper.start()
    try:
        yield
    finally:
        await reaper.stop()


app = FastAPI(title="bro-agent", lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# Bound absurd inputs before they touch the model or the DB.
MAX_MESSAGE_CHARS = 20_000

# SSE heartbeat: while a provider turn blocks (Horde queue waits can run
# minutes), emit a ping this often so the stream never goes quiet long
# enough for a proxy to kill it — and the UI can show it's still alive.
CHAT_HEARTBEAT_S = 20.0


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    """Structured per-request log: method, path, status, ms. No bodies."""
    import uuid as _uuid

    rid = _uuid.uuid4().hex[:8]
    start = time.monotonic()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        ms = (time.monotonic() - start) * 1000
        logger.info("req=%s %s %s -> %s %.1fms", rid, request.method, request.url.path, status, ms)
    response.headers["X-Request-ID"] = rid
    return response


def _access_code() -> str:
    return os.environ.get("BRO_ACCESS_CODE", "").strip()


def _check_access(provided: str) -> None:
    required = _access_code()
    if required and provided != required:
        raise HTTPException(status_code=401, detail="wrong access code")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = ""
    access_code: str = ""


class TrainCreateRequest(BaseModel):
    model_repo: str
    dataset: str = ""
    lora_rank: int = 16
    access_code: str = ""


class TrainClaimRequest(BaseModel):
    access_code: str = ""


class TrainCompleteRequest(BaseModel):
    access_code: str = ""
    status: str = ""
    result_url: str = ""
    metrics: dict = {}
    error: str = ""


@app.get("/api/health")
def health():
    router = get_router()
    return {
        "ok": True,
        "model": "horde",
        "horde_primary": True,
        "pollinations_fallback": True,
        "venice_removed": True,
        "providers": router.status(),
        "composio_configured": bool(os.environ.get("COMPOSIO_API_KEY", "").strip()),
        "access_gate": bool(_access_code()),
    }


@app.get("/api/models")
def models(access_code: str = ""):
    """Provider chain status: health, failure counts, cooldowns, Horde models."""
    _check_access(access_code)
    router = get_router()
    horde_models: list[str] = []
    try:
        from .horde_llm import HORDE_UNCENSORED_MODELS, resolve_models

        horde_models = resolve_models(HORDE_UNCENSORED_MODELS)
    except Exception as e:  # langchain_core missing in dep-free envs
        logger.warning("horde model list unavailable: %s", e)

    return {
        "order": router.preferred_order(),
        "providers": router.status(),
        "horde_models": horde_models,
    }


@app.get("/api/stats")
def usage_stats(access_code: str = "", days: int = 14):
    """Per-day chat turns per provider (counts only, no message content)."""
    _check_access(access_code)
    return {"stats": stats.get_stats(days)}


@app.get("/api/threads/{thread_id}/export")
def thread_export(thread_id: str, access_code: str = ""):
    """Download a thread as Markdown."""
    from fastapi.responses import PlainTextResponse

    _check_access(access_code)
    if not memory.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="thread not found")
    # Page through the whole thread: the old single fetch (limit=1000)
    # silently dropped every message past the 1000th — i.e. exactly the
    # recent conversation an export is for. Keep pages under the endpoint's
    # 1000-message clamp and stop on a short page.
    msgs: list[dict] = []
    offset = 0
    while True:
        page = memory.get_thread_messages(thread_id, limit=500, offset=offset)
        if not page:
            break
        msgs.extend(page)
        offset += len(page)
        if len(page) < 500:
            break
    lines = [f"# bro chat export", ""]
    for m in msgs:
        who = "Jay" if m["role"] == "user" else "bro"
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(m["created_at"])) if m.get("created_at") else ""
        lines.append(f"## {who}" + (f" ({ts})" if ts else ""))
        lines.append(m["content"])
        lines.append("")
    return PlainTextResponse(
        "\n".join(lines),
        headers={"Content-Disposition": f"attachment; filename=bro-chat-{thread_id}.md"},
    )


@app.post("/api/train/{job_id}/retry")
def train_retry(job_id: str, req: TrainClaimRequest):
    """Re-queue a failed train job (attempts+1, capped)."""
    _check_access(req.access_code)
    if not train_jobs.retry_job(job_id):
        raise HTTPException(status_code=404, detail="job not found, not failed, or out of attempts")
    return {"ok": True}


@app.post("/api/train/reap-stuck")
def train_reap_stuck(req: TrainClaimRequest):
    """Re-queue 'working' jobs whose worker vanished (no worker row needed).

    Returns {"requeued": [...], "failed": [...]}: a stuck job at the
    attempts cap is failed, not requeued, so a poison job retires instead
    of cycling claim -> stuck -> requeue forever (see
    train.fail_stuck_jobs)."""
    _check_access(req.access_code)
    return train_jobs.fail_stuck_jobs()


UPLOAD_MAX_BYTES = 50 * 1024 * 1024


def _adapter_max_bytes() -> int:
    """Per-file cap for train adapter uploads. Default 2 GB: a rank-16
    adapter is ~100 MB, but the max rank-256 job can legitimately exceed
    1 GB, so the cap must clear that while still bounding a runaway or
    hostile upload on bro-agent's ephemeral disk. Override with
    BRO_ADAPTER_MAX_BYTES (bytes) for bigger jobs or tighter disks."""
    try:
        return max(1, int(os.environ.get("BRO_ADAPTER_MAX_BYTES", 2 * 1024**3)))
    except ValueError:
        return 2 * 1024**3


ADAPTER_MAX_BYTES = _adapter_max_bytes()


def _adapter_job_max_bytes() -> int:
    """Total adapter bytes allowed per job. Default 4 GB: the per-file cap
    alone never bounded a job — a worker (or a bad adapter loop) could
    upload unlimited 2 GB files to one job and fill the disk. 4 GB covers
    one max-size adapter plus shards, and never sits below the per-file
    cap, so a single legit max-size file always fits. Override with
    BRO_ADAPTER_JOB_MAX_BYTES (bytes). Read per request so tests and
    deploys can tighten it without a restart."""
    try:
        env = os.environ.get("BRO_ADAPTER_JOB_MAX_BYTES", "")
        if env.strip():
            return max(1, int(env))
    except ValueError:
        pass
    return max(4 * 1024**3, ADAPTER_MAX_BYTES)


def _adapter_total_max_bytes() -> int:
    """Total adapter bytes allowed across ALL jobs. Default 20 GB: the
    per-job cap alone never bounded the tree — every job could hold its
    own 4 GB, so N jobs filled the disk N times over. 20 GB clears five
    max-size jobs and never sits below the per-job cap, so one legit
    max-size job always fits. Override with BRO_ADAPTER_TOTAL_MAX_BYTES
    (bytes). Read per request so tests and deploys can tighten it
    without a restart."""
    try:
        env = os.environ.get("BRO_ADAPTER_TOTAL_MAX_BYTES", "")
        if env.strip():
            return max(1, int(env))
    except ValueError:
        pass
    return max(20 * 1024**3, _adapter_job_max_bytes())


def _adapter_dest_is_symlink(dest: str) -> bool:
    """True if the file itself or its job dir is a symlink. Adapter files
    are only ever created as regular files by the upload route, so any
    symlink here is planted, not legit — refuse it instead of following it
    to a target outside the adapters tree."""
    return os.path.islink(dest) or os.path.islink(os.path.dirname(dest))


def _safe_upload_name(name: str) -> str:
    """Same sanitizer as the upload route — names the file can actually have."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in (name or "file"))[:128]
    return safe.lstrip(".") or "file"


@app.get("/api/uploads")
def uploads_list(access_code: str = ""):
    """List files in the agent's workdir/uploads/ (name, bytes, mtime)."""
    from .sandbox import resolve, workdir

    _check_access(access_code)
    root = workdir()
    uploads_dir = resolve(root, "uploads")
    items: list[dict] = []
    if os.path.isdir(uploads_dir):
        for entry in sorted(os.listdir(uploads_dir)):
            path = os.path.join(uploads_dir, entry)
            if os.path.isfile(path):
                st = os.stat(path)
                items.append({"name": entry, "bytes": st.st_size, "mtime": st.st_mtime})
    return {"uploads": items}


@app.delete("/api/uploads/{name}")
def upload_delete(name: str, access_code: str = ""):
    """Delete one uploaded file by name."""
    from .sandbox import resolve, workdir

    _check_access(access_code)
    root = workdir()
    safe_name = _safe_upload_name(name)
    dest = resolve(root, os.path.join("uploads", safe_name))
    if not os.path.isfile(dest):
        raise HTTPException(status_code=404, detail="upload not found")
    os.unlink(dest)
    return {"ok": True, "name": safe_name}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), access_code: str = Form("")):
    """Upload a file into the agent's workdir (workdir/uploads/)."""
    from .sandbox import display_path, resolve, workdir

    _check_access(access_code)
    root = workdir()
    safe_name = _safe_upload_name(file.filename)
    dest = resolve(root, os.path.join("uploads", safe_name))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > UPLOAD_MAX_BYTES:
                f.close()
                os.unlink(dest)
                raise HTTPException(status_code=413, detail="file too large (50MB cap)")
            f.write(chunk)
    return {"ok": True, "path": display_path(root, dest), "bytes": size}


@app.get("/api/threads")
def threads(access_code: str = "", search: str = ""):
    _check_access(access_code)
    if (search or "").strip():
        return {"threads": memory.search_threads(search)}
    return {"threads": memory.list_threads()}


@app.get("/api/threads/{thread_id}")
def thread_messages(thread_id: str, access_code: str = "", limit: int = 200, offset: int = 0):
    _check_access(access_code)
    if not memory.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="thread not found")
    return {"messages": memory.get_thread_messages(thread_id, limit=min(max(1, limit), 1000), offset=offset)}


class ThreadRenameRequest(BaseModel):
    title: str = ""
    access_code: str = ""


@app.post("/api/threads/{thread_id}/rename")
def thread_rename(thread_id: str, req: ThreadRenameRequest):
    _check_access(req.access_code)
    if not memory.rename_thread(thread_id, req.title):
        raise HTTPException(status_code=404, detail="thread not found or empty title")
    return {"ok": True}


class ThreadDeleteRequest(BaseModel):
    access_code: str = ""


@app.delete("/api/threads/{thread_id}")
def thread_delete(thread_id: str, access_code: str = ""):
    _check_access(access_code)
    if not memory.delete_thread(thread_id):
        raise HTTPException(status_code=404, detail="thread not found")
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    _check_access(req.access_code)
    # Rate limit: per access code when the gate is on, else per client IP.
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"code:{req.access_code}" if _access_code() else f"ip:{client_ip}"
    try:
        rate_limit.check(rate_key)
    except rate_limit.RateLimited as e:
        raise HTTPException(
            status_code=429,
            detail="too many requests",
            headers={"Retry-After": str(int(e.retry_after_s) + 1)},
        )
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="empty message")
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"message too long ({MAX_MESSAGE_CHARS} char cap)")

    # Hard-boundary pre-check (surgical guards only — legit work always passes).
    # Per-deployment refusal policy (app/guard_policy.py) runs second: this
    # instance's own owner-written patterns. Ships empty — nothing extra
    # refused until the owner adds an entry. The hard floor above is
    # separate and always applies first.
    hit = check_hard_boundary(message) or check_owner_boundary(message)
    if hit:
        thread_id = req.thread_id or memory.create_thread(message[:40])
        memory.add_message(thread_id, "user", message)
        reply = f"{hit.reason} {hit.alternative}"
        memory.add_message(thread_id, "assistant", reply)
        return {"thread_id": thread_id, "reply": reply, "blocked": True}

    thread_id = req.thread_id if req.thread_id and memory.thread_exists(req.thread_id) else memory.create_thread(message[:40])
    memory.add_message(thread_id, "user", message)
    history = memory.get_history(thread_id)

    async def stream():
        from .agent import get_cached_agent
        from .chat_stream import stream_agent_turn

        # The turn runs as a background task feeding an event queue, so the
        # generator can emit progress events and heartbeat pings while a
        # provider blocks (Horde queue waits can run minutes). Two wins: the
        # UI shows live status instead of a dead "thinking…" indicator, and
        # quiet SSE streams don't get killed by intermediate proxies.
        queue: asyncio.Queue = asyncio.Queue()

        async def run_turn() -> None:
            try:
                await queue.put({"type": "thread", "thread_id": thread_id})
                router = get_router()
                # Chat uses streaming-first order, not the router's
                # uncensored-first order: Horde queues with no token stream,
                # so Horde-first shows nothing until the job finishes (up to
                # the 60s cap), while Pollinations streams visibly in
                # seconds. Horde stays as the fallback below.
                order = router.chat_order()
                messages = [{"role": m["role"], "content": m["content"]} for m in build_window(history)]
                final_text, final_source, final_error = "", "", ""
                # Walk the provider chain: on hard error OR refusal boilerplate
                # from ANY model (Jay's standing order — never show Jay a
                # refusal on legit work while another provider is untried),
                # move to the next provider instead of giving up. A refusal
                # surfaces only when every provider in the chain refused.
                for i, provider in enumerate(order):
                    await queue.put({"type": "progress", "stage": "trying", "provider": provider})
                    try:
                        # Cached agent: built once per provider, reused across
                        # turns (see app/agent.py) instead of rebuilt per turn.
                        agent, source, _cache_hit = await get_cached_agent(preferred=provider)
                        # Bound the turn: a hung provider must not hang the stream forever.
                        turn_timeout = float(os.environ.get("BRO_TURN_TIMEOUT_S", "600"))
                        text = await asyncio.wait_for(
                            stream_agent_turn(agent, messages, queue.put),
                            timeout=turn_timeout,
                        )
                        if looks_like_refusal(text) and i < len(order) - 1:
                            logger.warning("provider %s refused legit work, skipping to next", source)
                            router.record_failure(source)
                            await queue.put(
                                {"type": "progress", "stage": "refused", "provider": source}
                            )
                            continue
                        router.record_success(source)
                        final_text, final_source = text, source
                        break
                    except Exception as e:
                        logger.exception("agent turn failed on %s", provider)
                        router.record_failure(provider)
                        await queue.put(
                            {
                                "type": "progress",
                                "stage": "failed",
                                "provider": provider,
                                "error": str(e)[:200],
                            }
                        )
                        final_error = str(e)
                if final_text:
                    memory.add_message(thread_id, "assistant", final_text)
                    in_chars = sum(len(str(m.get("content", ""))) for m in messages)
                    stats.record_turn(final_source, in_chars, len(final_text))
                    await queue.put({"type": "done", "reply": final_text, "model": final_source})
                else:
                    err = f"Agent error: {final_error or 'all providers failed'}"
                    memory.add_message(thread_id, "assistant", err)
                    await queue.put({"type": "error", "error": final_error or "all providers failed"})
            except Exception as e:  # never leave the consumer hanging on pings
                logger.exception("chat turn producer crashed")
                await queue.put({"type": "error", "error": f"internal error: {e}"})

        task = asyncio.create_task(run_turn())
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=CHAT_HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("type") in ("done", "error"):
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---- Pull-based GPU training jobs (train-studio system, inside bro-agent) ----


@app.post("/api/train")
def train_create(req: TrainCreateRequest):
    _check_access(req.access_code)
    model_repo = (req.model_repo or "").strip()
    if not model_repo:
        raise HTTPException(status_code=400, detail="model_repo is required")
    job_id = train_jobs.create_job(model_repo, (req.dataset or "").strip(), req.lora_rank)
    return {"job_id": job_id}


@app.get("/api/train/list")
def train_list(access_code: str = ""):
    _check_access(access_code)
    return {"jobs": train_jobs.list_jobs()}


@app.get("/api/train/{job_id}")
def train_get(job_id: str, access_code: str = ""):
    _check_access(access_code)
    job = train_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/api/train/claim")
def train_claim(req: TrainClaimRequest, request: Request):
    _check_access(req.access_code)
    # Workers poll every ~30s; bound it so a hot loop can't hammer the DB.
    ip = request.client.host if request.client else "unknown"
    try:
        rate_limit.check(f"claim:{ip}")
    except rate_limit.RateLimited:
        raise HTTPException(status_code=429, detail="claim too often", headers={"Retry-After": "5"})
    job = train_jobs.claim_job()
    if not job:
        raise HTTPException(status_code=404, detail="no queued jobs")
    return job


@app.post("/api/train/{job_id}/complete")
def train_complete(job_id: str, req: TrainCompleteRequest):
    _check_access(req.access_code)
    if req.status not in ("done", "failed"):
        raise HTTPException(status_code=400, detail="status must be done or failed")
    ok = train_jobs.complete_job(job_id, req.status, req.result_url, req.metrics or None, req.error)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True}


@app.post("/api/train/{job_id}/upload")
async def train_upload(job_id: str, file: UploadFile = File(...), access_code: str = Form("")):
    _check_access(access_code)
    if not train_jobs.get_job(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    try:
        dest = train_jobs.adapter_file_path(job_id, file.filename or "adapter.bin", create_dir=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="adapter path refused")
    if _adapter_dest_is_symlink(dest):
        raise HTTPException(status_code=400, detail="adapter path is a symlink, refused")
    # The generic /api/upload route caps at 50 MB, but this route used to
    # stream unbounded — one runaway worker (or bad adapter) could fill
    # bro-agent's ephemeral disk. Same pattern as /api/upload: 413 and
    # remove the partial file, so a rejected upload leaves nothing behind.
    # The per-file cap alone did not bound the job: unlimited files per
    # job still filled the disk, so the job total is capped too (a
    # re-upload of the same name replaces its old bytes, it does not
    # double-count them).
    #
    # Atomicity, 2026-10-31: the old code streamed straight into the
    # final destination, which had two consequences. A re-upload that
    # tripped a cap truncated the old file first, then the 413 handler
    # unlinked it — the previously good adapter was destroyed, not just
    # the rejected bytes. And a download running mid-upload served the
    # truncated, half-written file as a complete adapter. Now the upload
    # streams into an unguessable temp file in the same job dir and only
    # renames it into place once the whole upload fits under both
    # caps, so a rejected upload leaves the old file untouched and a
    # download never sees a partial file.
    # Pinned job dir, 2026-11-04: the temp file used to be created by
    # path (open(tmp) in the job dir) and renamed by path
    # (os.replace(tmp, dest)), and the symlink re-check just before the
    # replace only inspected the path at that instant. A job dir swapped
    # for a symlink after adapter_file_path()'s check — or after that
    # re-check — made BOTH operations resolve through the link, so the
    # temp file was created, and the adapter landed, in the link target
    # outside the adapters tree (the old "replace swaps the link, never
    # its target" reasoning held for a symlinked FILE, not for a
    # symlinked DIR, where both paths resolve through the link). Now
    # the route opens the job dir once with O_DIRECTORY | O_NOFOLLOW
    # (train.open_adapter_job_dir) and creates, writes, unlinks, and
    # renames strictly relative to that fd: a path swap after the open
    # cannot redirect any of them, because none of them re-resolves the
    # path. The temp create additionally uses O_CREAT | O_EXCL |
    # O_NOFOLLOW, so a temp name planted as a symlink fails instead of
    # being followed. The final re-check still refuses a symlinked
    # destination entry — but even without it, the rename would swap
    # the entry itself, never its target.
    # Serialization, 2026-11-01: the cap check below snapshots the job's
    # stored bytes before streaming, so two concurrent uploads to one job
    # used to both fit on their own numbers and land over the cap
    # together. The per-job lock makes the snapshot, the stream, and the
    # replace atomic with respect to other uploads to the same job, so a
    # concurrent second upload sees the first one's landed bytes and 413s
    # exactly as a sequential one would. Uploads to different jobs are
    # unaffected — the lock is per job, not global.
    # Aggregate cap, 2026-11-02: the per-job cap never bounded the tree —
    # every job could hold its own 4 GB, so uploads to many jobs filled
    # the disk job by job. The total cap below is checked the same way,
    # against a committed snapshot plus the in-flight reservation ledger
    # (train.py): a concurrent upload to another job reserves its
    # streamed bytes as it goes, so its not-yet-landed bytes already
    # count here, and two jobs racing the total cap cannot both land
    # over it. A re-upload replaces its old file, so the old bytes are
    # subtracted from the snapshot exactly as for the per-job cap.
    async with train_jobs.adapter_upload_lock(job_id):
        job_max = _adapter_job_max_bytes()
        total_max = _adapter_total_max_bytes()
        base_total = train_jobs.adapter_total_bytes(job_id)
        global_base = train_jobs.adapters_total_bytes()
        # Pin the job dir before touching anything inside it (see the
        # pinned-dir comment above): every create, stat, unlink, and
        # rename below is relative to this fd, never to a path, so a
        # job-dir symlink swapped in after this point cannot redirect
        # the upload outside the adapters tree. A swap that already
        # happened fails here (O_NOFOLLOW) instead of being followed.
        try:
            dir_fd = train_jobs.open_adapter_job_dir(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="adapter path is a symlink, refused")
        safe_name = os.path.basename(dest)
        try:
            old_size = 0
            try:
                dest_st = os.stat(safe_name, dir_fd=dir_fd, follow_symlinks=False)
            except OSError:
                dest_st = None
            if dest_st is not None:
                if stat.S_ISLNK(dest_st.st_mode) or not stat.S_ISREG(dest_st.st_mode):
                    raise HTTPException(status_code=400, detail="adapter path is a symlink, refused")
                old_size = dest_st.st_size
                base_total = max(0, base_total - old_size)
            tmp_name = f".upload-{uuid.uuid4().hex}.tmp"
            size = 0
            token = uuid.uuid4().hex
            train_jobs.reserve_upload(token, 0)
            try:
                try:
                    # O_EXCL: a temp name planted as a symlink (or any
                    # pre-existing entry) fails instead of being
                    # followed; O_NOFOLLOW is belt-and-braces on top.
                    tmp_fd = os.open(
                        tmp_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=dir_fd,
                    )
                    with os.fdopen(tmp_fd, "wb") as f:
                        while True:
                            chunk = await file.read(1024 * 1024)
                            if not chunk:
                                break
                            size += len(chunk)
                            train_jobs.reserve_upload(token, size)
                            if (
                                size > ADAPTER_MAX_BYTES
                                or base_total + size > job_max
                                or global_base - old_size + train_jobs.inflight_reserved_bytes() > total_max
                            ):
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"adapter file too large ({ADAPTER_MAX_BYTES} byte per-file cap, {job_max} byte per-job cap, {total_max} byte total cap; set BRO_ADAPTER_MAX_BYTES / BRO_ADAPTER_JOB_MAX_BYTES / BRO_ADAPTER_TOTAL_MAX_BYTES to raise them)",
                                )
                            f.write(chunk)
                except BaseException:
                    # Rejected or aborted upload: drop the temp file, leave whatever
                    # was at dest (including a good previous adapter) exactly as is.
                    try:
                        os.unlink(tmp_name, dir_fd=dir_fd)
                    except OSError:
                        pass
                    raise
                # Final re-check under the cross-job commit lock: the
                # per-chunk check above reads a snapshot from upload start,
                # so an upload that landed and released mid-stream would be
                # invisible to it. Re-scanning here catches that case, and
                # the lock keeps two racing uploads from both passing and
                # both landing. A 413 here still drops only the temp file.
                # The destination entry is re-checked relative to the
                # pinned dir fd, and the rename itself is dir-fd to
                # dir-fd (POSIX rename replaces the entry, never
                # follows a symlinked one), so a swap between this
                # check and the rename is still safe.
                with train_jobs.adapter_commit_lock():
                    try:
                        fresh_st = os.stat(safe_name, dir_fd=dir_fd, follow_symlinks=False)
                    except OSError:
                        fresh_st = None
                    if fresh_st is not None and (
                        stat.S_ISLNK(fresh_st.st_mode) or not stat.S_ISREG(fresh_st.st_mode)
                    ):
                        try:
                            os.unlink(tmp_name, dir_fd=dir_fd)
                        except OSError:
                            pass
                        raise HTTPException(status_code=400, detail="adapter path is a symlink, refused")
                    fresh_total = train_jobs.adapters_total_bytes()
                    if fresh_total - old_size + train_jobs.inflight_reserved_bytes() > total_max:
                        try:
                            os.unlink(tmp_name, dir_fd=dir_fd)
                        except OSError:
                            pass
                        raise HTTPException(
                            status_code=413,
                            detail=f"adapter file too large ({ADAPTER_MAX_BYTES} byte per-file cap, {job_max} byte per-job cap, {total_max} byte total cap; set BRO_ADAPTER_MAX_BYTES / BRO_ADAPTER_JOB_MAX_BYTES / BRO_ADAPTER_TOTAL_MAX_BYTES to raise them)",
                        )
                    os.rename(tmp_name, safe_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            finally:
                train_jobs.release_upload(token)
        finally:
            os.close(dir_fd)
    return {"ok": True, "filename": os.path.basename(dest)}


@app.get("/api/train/{job_id}/files/{filename}")
def train_file(job_id: str, filename: str, access_code: str = ""):
    _check_access(access_code)
    # Served from an open file descriptor, not by path (see
    # train.open_adapter_file): the old code checked islink/isfile on
    # the path and then handed that path to FileResponse, which opened
    # it later — a symlink swapped in between the check and the open
    # made the route serve the link target outside the adapters tree.
    # The fd pins the inode, so a swap after the open cannot change
    # what is served, and an os.replace() mid-download serves the
    # complete old adapter instead of a half-written new one.
    try:
        fd, size, safe_name = train_jobs.open_adapter_file(job_id, filename)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="file not found")

    def _stream():
        try:
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            os.close(fd)

    return StreamingResponse(
        _stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(size),
            "Content-Disposition": f'attachment; filename="{safe_name}"',
        },
    )


class ForgeWorkerRegisterRequest(BaseModel):
    platform: str = ""
    access_code: str = ""


class ForgeWorkerHeartbeatRequest(BaseModel):
    worker_id: str = ""
    current_job_id: str = ""
    access_code: str = ""


# ---- Forge worker dispatcher (Colab + Kaggle registry, heartbeats, reaper) ----


@app.post("/api/forge/workers/register")
def forge_worker_register(req: ForgeWorkerRegisterRequest):
    _check_access(req.access_code)
    worker_id = forge_dispatch.register_worker(req.platform)
    return {"worker_id": worker_id}


@app.post("/api/forge/workers/heartbeat")
def forge_worker_heartbeat(req: ForgeWorkerHeartbeatRequest):
    _check_access(req.access_code)
    ok = forge_dispatch.heartbeat(req.worker_id, req.current_job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="unknown worker_id")
    return {"ok": True}


@app.post("/api/forge/workers/{worker_id}/finished")
def forge_worker_finished(worker_id: str, req: ForgeWorkerHeartbeatRequest):
    _check_access(req.access_code)
    forge_dispatch.job_finished(worker_id)
    forge_dispatch.heartbeat(worker_id, "")
    return {"ok": True}


@app.get("/api/forge/workers")
def forge_workers(access_code: str = ""):
    _check_access(access_code)
    snap = forge_dispatch.status_snapshot()
    snap["reaper"] = reaper.status()
    return snap


@app.post("/api/forge/dispatch/reap")
def forge_dispatch_reap(req: ForgeWorkerHeartbeatRequest):
    _check_access(req.access_code)
    return forge_dispatch.reap_dead_workers()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
