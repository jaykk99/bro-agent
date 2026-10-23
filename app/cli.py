"""bro terminal chat — talk to bro-agent from the command line.

Usage:
    BRO_BASE_URL=https://bro-agent.onrender.com BRO_ACCESS_CODE=... python -m app.cli

Streams the reply as it arrives (SSE). Commands inside the REPL:
    /new            start a new thread
    /threads        list recent threads
    /thread <id>    switch to a thread
    /rename <title> rename the current thread
    /delete         delete the current thread (asks first)
    /export [path]  download the current thread as Markdown
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


def rename_thread(thread_id: str, title: str) -> bool:
    """Rename a thread. Returns True on success."""
    if not thread_id or not (title or "").strip():
        return False
    r = httpx.post(
        f"{BASE}/api/threads/{thread_id}/rename",
        json={"title": title.strip()[:60], "access_code": CODE},
        timeout=20,
    )
    return r.status_code == 200


def delete_thread(thread_id: str) -> bool:
    """Delete a thread. Returns True on success."""
    if not thread_id:
        return False
    r = httpx.delete(f"{BASE}/api/threads/{thread_id}", params=_q({}), timeout=20)
    return r.status_code == 200


def export_thread(thread_id: str, dest: str = "") -> str:
    """Download a thread as Markdown and save it. Returns the file path."""
    if not thread_id:
        raise ValueError("no thread selected")
    r = httpx.get(f"{BASE}/api/threads/{thread_id}/export", params=_q({}), timeout=30)
    r.raise_for_status()
    path = dest or f"bro-chat-{thread_id[:8]}.md"
    with open(path, "wb") as f:
        f.write(r.content)
    return path


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
                    elif evt.get("type") == "progress":
                        # Live status while the turn runs (provider hops).
                        sys.stdout.write(
                            f"\r\033[Kbro> ({evt.get('stage', '?')} {evt.get('provider', '?')}…)"
                        )
                        sys.stdout.flush()
                    elif evt.get("type") == "ping":
                        pass  # heartbeat — the turn is still alive, keep the status line
                    elif evt.get("type") == "done":
                        reply = evt.get("reply", "")
                        model = evt.get("model", "")
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                    elif evt.get("type") == "error":
                        reply = "ERR: " + str(evt.get("error", "unknown"))
    return tid, reply, model


def main() -> None:
    print(f"bro @ {BASE} — /new, /threads, /thread <id>, /rename <title>, /delete, /export [path], /quit")
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
        if line.startswith("/rename"):
            title = line[len("/rename"):].strip()
            if not thread_id:
                print("(no thread selected)")
            elif not title:
                print("usage: /rename <title>")
            elif rename_thread(thread_id, title):
                print(f"(renamed to {title[:60]})")
            else:
                print("(rename failed)")
            continue
        if line == "/delete":
            if not thread_id:
                print("(no thread selected)")
                continue
            sure = input("delete this thread? [y/N] ").strip().lower()
            if sure in ("y", "yes") and delete_thread(thread_id):
                print("(deleted)")
                thread_id = ""
            else:
                print("(kept)")
            continue
        if line.startswith("/export"):
            if not thread_id:
                print("(no thread selected)")
            else:
                dest = line[len("/export"):].strip()
                try:
                    path = export_thread(thread_id, dest)
                    print(f"(exported to {path})")
                except Exception as e:
                    print(f"(export failed: {e})")
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
