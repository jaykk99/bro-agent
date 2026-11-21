"""CLI streaming + REPL tests — chat_stream SSE parsing and main() commands.

Companion to tests/test_cli.py (which covers rename/delete/export
endpoints). Together they pin the terminal client end to end without
touching the network: httpx and input() are both faked.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.cli as cli  # noqa: E402


# ---- fakes ----


class _FakeSSE:
    """httpx.stream() context manager yielding canned SSE chunks."""

    def __init__(self, status_code=200, chunks=(), raise_status=None):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._raise_status = raise_status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self._raise_status is not None:
            raise self._raise_status

    def iter_text(self):
        yield from self._chunks


class _FakeHttpxStream:
    """Fake httpx with a stream() method; records the chat POST."""

    def __init__(self, sse: _FakeSSE):
        self.sse = sse
        self.stream_calls: list[tuple] = []

    def stream(self, method, url, **kwargs):
        self.stream_calls.append((method, url, kwargs))
        return self.sse


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _install_stream(monkeypatch, sse: _FakeSSE):
    fake = _FakeHttpxStream(sse)
    monkeypatch.setattr(cli, "httpx", fake)
    monkeypatch.setattr(cli, "BASE", "http://test")
    monkeypatch.setattr(cli, "CODE", "secret")
    return fake


def _run_main(monkeypatch, inputs, threads=None, **stubs):
    """Run cli.main() with faked input() and stubbed helpers.

    `inputs` is consumed in order; EOFError ends the loop like Ctrl-D.
    Returns (printed_stdout, stub_call_log)."""
    it = iter(inputs)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError()  # Ctrl-D at the prompt

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(cli, "list_threads", lambda: list(threads or []))
    log: dict = {}

    def _stub(name, fn):
        def wrapper(*a, **k):
            log.setdefault(name, []).append((a, k))
            return fn(*a, **k)

        monkeypatch.setattr(cli, name, wrapper)

    _stub("chat_stream", stubs.get("chat_stream", lambda msg, tid: (tid, "stub reply", "horde")))
    _stub("rename_thread", stubs.get("rename_thread", lambda tid, t: True))
    _stub("delete_thread", stubs.get("delete_thread", lambda tid: True))
    _stub("export_thread", stubs.get("export_thread", lambda tid, dest="": "/tmp/x.md"))
    monkeypatch.setattr(cli, "BASE", "http://test")
    monkeypatch.setattr(cli, "CODE", "")
    import io

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    cli.main()
    return out.getvalue(), log


# ---- chat_stream: SSE parsing ----


def test_chat_stream_401(monkeypatch):
    _install_stream(monkeypatch, _FakeSSE(status_code=401))
    tid, reply, model = cli.chat_stream("hi", "t1")
    assert (tid, reply, model) == ("t1", "", "ERR: wrong access code")


def test_chat_stream_429(monkeypatch):
    _install_stream(monkeypatch, _FakeSSE(status_code=429))
    tid, reply, model = cli.chat_stream("hi", "t1")
    assert model == "ERR: rate limited — slow down a sec"
    assert reply == ""


def test_chat_stream_done_and_thread_events(monkeypatch):
    chunks = [
        _sse_event({"type": "thread", "thread_id": "new-id"})
        + _sse_event({"type": "progress", "stage": "trying", "provider": "horde"}),
        _sse_event({"type": "ping"}) + _sse_event({"type": "done", "reply": "hello!", "model": "horde"}),
    ]
    fake = _install_stream(monkeypatch, _FakeSSE(chunks=chunks))
    tid, reply, model = cli.chat_stream("hi", "")
    assert (tid, reply, model) == ("new-id", "hello!", "horde")
    method, url, kwargs = fake.stream_calls[0]
    assert method == "POST" and url == "http://test/api/chat"
    assert kwargs["json"] == {"message": "hi", "thread_id": "", "access_code": "secret"}


def test_chat_stream_error_event(monkeypatch):
    chunks = [_sse_event({"type": "error", "error": "boom"})]
    _install_stream(monkeypatch, _FakeSSE(chunks=chunks))
    _, reply, _ = cli.chat_stream("hi", "t1")
    assert reply == "ERR: boom"


def test_chat_stream_skips_bad_lines_and_split_chunks(monkeypatch):
    # Malformed JSON and non-data lines are skipped; an event split
    # across two iter_text() chunks is still parsed once complete.
    chunks = [
        "data: not json\n\n",
        ": a comment\n\ndata: {\"type\": \"do",
        "ne\", \"reply\": \"split ok\", \"model\": \"pollinations\"}\n\n",
    ]
    _install_stream(monkeypatch, _FakeSSE(chunks=chunks))
    _, reply, model = cli.chat_stream("hi", "t1")
    assert reply == "split ok" and model == "pollinations"


def test_chat_stream_http_error_raises(monkeypatch):
    _install_stream(monkeypatch, _FakeSSE(status_code=500, raise_status=RuntimeError("HTTP 500")))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        cli.chat_stream("hi", "t1")


# ---- list_threads / access-code query ----


def test_list_threads_returns_threads(monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            return {"threads": [{"id": "a", "title": "first"}]}

        def raise_for_status(self):
            return None

    calls = []

    class Httpx:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return Resp()

    monkeypatch.setattr(cli, "httpx", Httpx())
    monkeypatch.setattr(cli, "BASE", "http://test")
    monkeypatch.setattr(cli, "CODE", "secret")
    assert cli.list_threads() == [{"id": "a", "title": "first"}]
    assert calls[0][1]["params"] == {"access_code": "secret"}


def test_q_omits_access_code_when_unset(monkeypatch):
    monkeypatch.setattr(cli, "CODE", "")
    assert cli._q({}) == {}
    monkeypatch.setattr(cli, "CODE", "secret")
    assert cli._q({}) == {"access_code": "secret"}


def test_export_default_filename(monkeypatch, tmp_path):
    body = b"# export\n"

    class Resp:
        status_code = 200
        content = body

        def raise_for_status(self):
            return None

    class Httpx:
        def get(self, url, **kwargs):
            return Resp()

    monkeypatch.setattr(cli, "httpx", Httpx())
    monkeypatch.setattr(cli, "BASE", "http://test")
    monkeypatch.setattr(cli, "CODE", "")
    monkeypatch.chdir(tmp_path)
    path = cli.export_thread("abcdef123456")
    assert path == "bro-chat-abcdef12.md"
    assert (tmp_path / path).read_bytes() == body


# ---- main() REPL commands ----


def test_main_quits_and_resumes_latest(monkeypatch):
    out, _ = _run_main(monkeypatch, ["/quit"], threads=[{"id": "t1", "title": "hello"}])
    assert "resumed latest thread: hello" in out
    assert "bro @ http://test" in out


def test_main_unreachable_server(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "/quit")
    monkeypatch.setattr(cli, "BASE", "http://test")
    monkeypatch.setattr(cli, "list_threads", lambda: (_ for _ in ()).throw(RuntimeError("conn refused")))
    import io

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    cli.main()
    assert "can't reach http://test" in out.getvalue()


def test_main_eof_exits(monkeypatch):
    out, _ = _run_main(monkeypatch, [])
    assert "bye" in out


def test_main_empty_line_skipped_then_chat(monkeypatch):
    out, log = _run_main(monkeypatch, ["", "hello", "/quit"])
    assert len(log["chat_stream"]) == 1
    assert "bro [horde]> stub reply" in out


def test_main_new_and_threads(monkeypatch):
    threads = [{"id": "abc123", "title": "first"}, {"id": "def456", "title": "second"}]
    out, _ = _run_main(monkeypatch, ["/new", "/threads", "/quit"], threads=threads)
    assert "(new thread)" in out
    assert "abc123" in out and "second" in out


def test_main_thread_switch_match_and_miss(monkeypatch):
    threads = [{"id": "abc123", "title": "first"}]
    out, _ = _run_main(monkeypatch, ["/thread abc", "/thread zzz", "/quit"], threads=threads)
    assert "(switched to first)" in out
    assert "(no such thread)" in out


def test_main_rename_paths(monkeypatch):
    # No thread selected, then no title, then success, then failure.
    out, log = _run_main(
        monkeypatch,
        ["/new", "/rename x", "/thread abc", "/rename", "/rename good", "/rename bad", "/quit"],
        threads=[{"id": "abc123", "title": "old"}],
        rename_thread=lambda tid, t: t == "good",
    )
    assert "(no thread selected)" in out
    assert "usage: /rename <title>" in out
    assert "(renamed to good)" in out
    assert "(rename failed)" in out
    assert len(log["rename_thread"]) == 2


def test_main_delete_confirm_yes(monkeypatch):
    out, log = _run_main(monkeypatch, ["/delete", "y", "/quit"], threads=[{"id": "t1", "title": "old"}])
    assert "(deleted)" in out
    assert len(log["delete_thread"]) == 1


def test_main_delete_confirm_no_keeps(monkeypatch):
    out, log = _run_main(monkeypatch, ["/delete", "n", "/quit"], threads=[{"id": "t1", "title": "old"}])
    assert "(kept)" in out
    assert log.get("delete_thread") is None


def test_main_delete_no_thread(monkeypatch):
    out, _ = _run_main(monkeypatch, ["/new", "/delete", "/quit"], threads=[])
    assert "(no thread selected)" in out


def test_main_export_paths(monkeypatch):
    out, log = _run_main(
        monkeypatch,
        ["/export", "/export /tmp/out.md", "/quit"],
        threads=[{"id": "t1", "title": "old"}],
        export_thread=lambda tid, dest="": (_ for _ in ()).throw(RuntimeError("disk full"))
        if dest == "/tmp/out.md"
        else "/tmp/x.md",
    )
    assert "(exported to /tmp/x.md)" in out
    assert "(export failed: disk full)" in out


def test_main_export_no_thread(monkeypatch):
    out, _ = _run_main(monkeypatch, ["/new", "/export", "/quit"], threads=[])
    assert "(no thread selected)" in out


def test_main_chat_error_printed(monkeypatch):
    def boom(msg, tid):
        raise RuntimeError("server down")

    out, _ = _run_main(monkeypatch, ["hello", "/quit"], threads=[], chat_stream=boom)
    assert "error: server down" in out
