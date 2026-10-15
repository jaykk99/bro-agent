"""bro-agent — Jay's personal uncensored agent app. FastAPI + UniHarness.

Endpoints:
  GET  /            — chat UI
  GET  /api/health  — liveness + model source
  GET  /api/threads — list threads
  POST /api/chat    — chat turn (SSE stream). Body: {message, thread_id?, access_code}
  POST /api/train   — queue a fine-tune job. Body: {model_repo, dataset?, lora_rank?, access_code}
  GET  /api/train/list — recent training jobs (access_code query param)
  GET  /api/train/{id} — one training job (access_code query param)
  POST /api/train/claim — worker claims oldest queued job. Body: {access_code}
  POST /api/train/{id}/complete — worker reports done/failed. Body: {access_code, status, ...}
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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import memory
from . import train as train_jobs
from . import dispatch as forge_dispatch
from .guards import check_hard_boundary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="bro-agent")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


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
    return {
        "ok": True,
        "model": "pollinations",
        "pollinations_primary": True,
        "horde_fallback": True,
        "venice_removed": True,
        "composio_configured": bool(os.environ.get("COMPOSIO_API_KEY", "").strip()),
        "access_gate": bool(_access_code()),
    }


@app.get("/api/threads")
def threads(access_code: str = ""):
    _check_access(access_code)
    return {"threads": memory.list_threads()}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    _check_access(req.access_code)
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="empty message")

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

        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"
        try:
            agent, source = await create_bro_agent()
            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            result = await agent.ainvoke({"messages": messages})
            out_messages = result["messages"] if isinstance(result, dict) else []
            text = ""
            if out_messages:
                last = out_messages[-1]
                text = getattr(last, "content", "") or str(last)
            memory.add_message(thread_id, "assistant", text)
            yield f"data: {json.dumps({'type': 'done', 'reply': text, 'model': source})}\n\n"
        except Exception as e:
            logger.exception("agent turn failed")
            err = f"Agent error: {e}"
            memory.add_message(thread_id, "assistant", err)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

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
def train_claim(req: TrainClaimRequest):
    _check_access(req.access_code)
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
