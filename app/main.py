"""bro-agent — Jay's personal uncensored agent app. FastAPI + UniHarness.

Endpoints:
  GET  /            — chat UI
  GET  /api/health  — liveness + model source
  GET  /api/threads — list threads
  POST /api/chat    — chat turn (SSE stream). Body: {message, thread_id?, access_code}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import memory
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


@app.get("/api/health")
def health():
    from .agent import _venice_key

    return {
        "ok": True,
        "model": "venice" if _venice_key() else "horde",
        "venice_configured": bool(_venice_key()),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
