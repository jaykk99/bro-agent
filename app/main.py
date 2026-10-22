"""bro-agent — Jay's personal uncensored agent app. FastAPI + UniHarness.

Endpoints:
  GET  /            — chat UI
  GET  /api/health  — liveness + model source + provider status
  GET  /api/models  — provider chain status (order, health, cooldowns)
  GET  /api/stats   — per-day/per-provider usage counts (access_code query param)
  GET  /api/threads — list threads
  GET  /api/threads/{id} — thread messages with timestamps
  POST /api/threads/{id}/rename — rename a thread. Body: {title, access_code}
  DELETE /api/threads/{id} — delete a thread and its messages
  GET  /api/threads/{id}/export — thread as a Markdown download
  POST /api/chat    — chat turn (SSE stream). Body: {message, thread_id?, access_code}
                     Events: thread, progress{stage: trying|failed|refused, provider},
                     ping (20s heartbeat), done{reply, model}, error
  POST /api/upload  — upload a file into workdir/uploads/. Form: {file, access_code}
  POST /api/train   — queue a fine-tune job. Body: {model_repo, dataset?, lora_rank?, access_code}
  GET  /api/train/list — recent training jobs (access_code query param)
  GET  /api/train/{id} — one training job (access_code query param)
  POST /api/train/claim — worker claims oldest queued job. Body: {access_code}
  POST /api/train/{id}/complete — worker reports done/failed. Body: {access_code, status, ...}
  POST /api/train/{id}/retry — re-queue a failed job (attempts+1, capped). Body: {access_code}
  POST /api/train/reap-stuck — re-queue 'working' jobs whose worker vanished. Body: {access_code}
  POST /api/train/{id}/upload — worker uploads adapter file (multipart). Form: {file, access_code}
  GET  /api/train/{id}/files/{name} — download an adapter file (access_code query param)
  POST /api/forge/workers/register — worker registers (platform colab/kaggle). Body: {platform, access_code}
  POST /api/forge/workers/heartbeat — worker heartbeat. Body: {worker_id, current_job_id?, access_code}
  POST /api/forge/workers/{id}/finished — worker reports job done (clears current job). Body: {access_code}
  GET  /api/forge/workers — dispatcher status: workers alive per platform + queue depth (access_code query param)
  POST /api/forge/dispatch/reap — reap dead workers, re-queue their stuck jobs. Body: {access_code}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import memory
from . import train as train_jobs
from . import dispatch as forge_dispatch
from .context import build_window
from .guards import check_hard_boundary
from . import rate_limit
from . import stats
from .router import get_router, looks_like_refusal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="bro-agent")

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
        "model": "pollinations",
        "pollinations_primary": True,
        "horde_fallback": True,
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
    msgs = memory.get_thread_messages(thread_id, limit=1000)
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
    """Re-queue 'working' jobs whose worker vanished (no worker row needed)."""
    _check_access(req.access_code)
    return {"requeued": train_jobs.fail_stuck_jobs()}


UPLOAD_MAX_BYTES = 50 * 1024 * 1024


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), access_code: str = Form("")):
    """Upload a file into the agent's workdir (workdir/uploads/)."""
    from .sandbox import display_path, resolve, workdir

    _check_access(access_code)
    root = workdir()
    safe_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in (file.filename or "file"))[:128]
    safe_name = safe_name.lstrip(".") or "file"  # no hidden files, no dot-only names
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
def threads(access_code: str = ""):
    _check_access(access_code)
    return {"threads": memory.list_threads()}


@app.get("/api/threads/{thread_id}")
def thread_messages(thread_id: str, access_code: str = ""):
    _check_access(access_code)
    if not memory.thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="thread not found")
    return {"messages": memory.get_thread_messages(thread_id)}


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
    hit = check_hard_boundary(message)
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
        from .agent import create_bro_agent

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
                order = router.preferred_order()
                messages = [{"role": m["role"], "content": m["content"]} for m in build_window(history)]
                final_text, final_source, final_error = "", "", ""
                # Walk the provider chain: on hard error OR refusal boilerplate
                # from a fallback model (Jay's standing order — never show Jay
                # a refusal on legit work), move to the next provider instead
                # of giving up.
                for i, provider in enumerate(order):
                    await queue.put({"type": "progress", "stage": "trying", "provider": provider})
                    try:
                        agent, source = await create_bro_agent(preferred=provider)
                        # Bound the turn: a hung provider must not hang the stream forever.
                        turn_timeout = float(os.environ.get("BRO_TURN_TIMEOUT_S", "600"))
                        result = await asyncio.wait_for(
                            agent.ainvoke({"messages": messages}), timeout=turn_timeout
                        )
                        out_messages = result["messages"] if isinstance(result, dict) else []
                        text = ""
                        if out_messages:
                            last = out_messages[-1]
                            text = getattr(last, "content", "") or str(last)
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
    dest = train_jobs.adapter_file_path(job_id, file.filename or "adapter.bin")
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return {"ok": True, "filename": os.path.basename(dest)}


@app.get("/api/train/{job_id}/files/{filename}")
def train_file(job_id: str, filename: str, access_code: str = ""):
    _check_access(access_code)
    dest = train_jobs.adapter_file_path(job_id, filename)
    if not os.path.isfile(dest):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(dest)


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
    return forge_dispatch.status_snapshot()


@app.post("/api/forge/dispatch/reap")
def forge_dispatch_reap(req: ForgeWorkerHeartbeatRequest):
    _check_access(req.access_code)
    return forge_dispatch.reap_dead_workers()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
