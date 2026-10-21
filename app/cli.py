"""bro terminal chat — talk to bro-agent from the command line.

Usage:
    BRO_BASE_URL=https://bro-agent.onrender.com BRO_ACCESS_CODE=... python -m app.cli

Streams the reply as it arrives (SSE). Commands inside the REPL:
    /new            start a new thread
    /threads        list recent threads
    /thread <id>    switch to a thread
    /quit           exit
"""

from __future__ import annotations

import json
import os
import sys

import httpx

BASE = os.environ.get("BRO_BASE_URL", "http://localhost:8000").rstrip("/")
CODE = os.environ.get("BRO_ACCESS_CODE", "")


def _q(params: dict) -> dict:
    if CODE:
        params["access_code"] = CODE
    return params


def list_threads() -> list[dict]:
    r = httpx.get(f"{BASE}/api/threads", params=_q({}), timeout=20)
    r.raise_for_status()
    return r.json().get("threads", [])


def chat_stream(message: str, thread_id: str) -> tuple[str, str, str]:
    """POST /api/chat and stream SSE. Returns (thread_id, reply, model)."""
    reply, model, tid = "", "", thread_id
    with httpx.stream(
        "POST",
        f"{BASE}/api/chat",
        json={"message": message, "thread_id": thread_id, "access_code": CODE},
        timeout=900,
    ) as r:
        if r.status_code == 401:
            return tid, "", "ERR: wrong access code"
        if r.status_code == 429:
            return tid, "", "ERR: rate limited — slow down a sec"
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                part, buf = buf.split("\n\n", 1)
                for line in part.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") == "thread" and evt.get("thread_id"):
                        tid = evt["thread_id"]
                    elif evt.get("type") == "done":
                        reply = evt.get("reply", "")
                        model = evt.get("model", "")
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                    elif evt.get("type") == "error":
                        reply = "ERR: " + str(evt.get("error", "unknown"))
    return tid, reply, model


def main() -> None:
    print(f"bro @ {BASE} — /new, /threads, /thread <id>, /quit")
    thread_id = ""
    try:
        threads = list_threads()
    except Exception as e:
        print(f"can't reach {BASE}: {e}")
        return
    if threads:
        thread_id = threads[0]["id"]
        print(f"(resumed latest thread: {threads[0].get('title', '')[:40]})")
    while True:
        try:
            line = input("\nJay> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue
        if line == "/quit":
            return
        if line == "/new":
            thread_id = ""
            print("(new thread)")
            continue
        if line == "/threads":
            for t in list_threads()[:15]:
                mark = "*" if t["id"] == thread_id else " "
                print(f"{mark} {t['id'][:8]}  {(t.get('title') or '')[:50]}")
            continue
        if line.startswith("/thread "):
            want = line.split(None, 1)[1]
            match = [t for t in list_threads() if t["id"].startswith(want)]
            if match:
                thread_id = match[0]["id"]
                print(f"(switched to {match[0].get('title', '')[:50]})")
            else:
                print("(no such thread)")
            continue
        try:
            sys.stdout.write("bro> ...")
            sys.stdout.flush()
            thread_id, reply, model = chat_stream(line, thread_id)
            tag = f" [{model}]" if model and not model.startswith("ERR") else ""
            print(f"\r\033[Kbro{tag}> {reply}")
        except Exception as e:
            print(f"\r\033[Kerror: {e}")


if __name__ == "__main__":
    main()
