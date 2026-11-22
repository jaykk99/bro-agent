"""Shell tools: output cap, run_python cleanup failure, and the
background-job paths test_exec.py leaves open (timeout kill,
Popen failure, kill-before-spawn, job-table eviction / overflow).

No network; local commands only. Background jobs are short-lived
(sleep with a 1s timeout, or killed immediately).
"""

import asyncio
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import tools_exec as te  # noqa: E402
from app.tools_exec import (  # noqa: E402
    MAX_OUTPUT_CHARS,
    BgKillTool,
    BgPollTool,
    RunPythonTool,
    RunShellTool,
    StartBgTool,
    _BgJob,
    _get,
    _jobs,
    _register,
    exec_tools,
)


@pytest.fixture()
def _workdir(tmp_path, monkeypatch):
    r = str(tmp_path / "work")
    os.makedirs(r)
    monkeypatch.setenv("BRO_WORKDIR", r)
    return r


@pytest.fixture(autouse=True)
def _clean_jobs():
    yield
    with te._jobs_lock:
        _jobs.clear()


def run(tool, **kwargs):
    return asyncio.run(tool(**kwargs)).output


def _wait_done(job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _get(job_id)
        if job is not None and not job.poll()["running"]:
            return job.poll()
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


# ---- run_shell / run_python ----------------------------------------------

def test_run_shell_output_capped(_workdir):
    out = run(RunShellTool(), command="python3 -c \"print('y'*20000)\"")
    assert f"[... output capped at {MAX_OUTPUT_CHARS} chars ...]" in out
    assert out.startswith("exit=0")


def test_run_python_blocked_pattern(_workdir):
    assert run(RunPythonTool(), code="rm -rf /").startswith("Blocked: destructive")


def test_run_python_tmp_cleanup_failure_ignored(_workdir, monkeypatch):
    real_unlink = os.unlink

    def flaky_unlink(path, *args, **kwargs):
        if ".bro-tmp-" in os.fspath(path):
            raise OSError("injected unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", flaky_unlink)
    out = run(RunPythonTool(), code="print('still works')")
    assert out == "exit=0\nstill works"
    leftovers = [n for n in os.listdir(_workdir) if n.startswith(".bro-tmp-")]
    assert len(leftovers) == 1, "script stays when cleanup fails, result unaffected"


# ---- background jobs: timeout, kill, failure ------------------------------

def test_bg_job_timeout_kills_and_marks_output(_workdir):
    out = run(StartBgTool(), command="sleep 30", timeout_s=1)
    jid = out.split("started job ")[1].split(":")[0]
    state = _wait_done(jid)
    assert state["returncode"] == -9, f"expected SIGKILL, got {state}"
    assert "[killed after 1s timeout]" in state["output_tail"]
    assert "EXITED code=-9" in run(BgPollTool(), job_id=jid)


def test_bg_kill_before_spawn_still_kills(_workdir):
    job = _BgJob("sleep 30", _workdir, 60)
    job.kill()  # lands before Popen is assigned; _run must honor the flag
    assert job._kill_requested is True
    job.start()
    job._thread.join(timeout=10)
    assert job.poll()["returncode"] == -9
    assert job.poll()["running"] is False


def test_bg_kill_live_process(_workdir):
    out = run(StartBgTool(), command="sleep 30", timeout_s=600)
    jid = out.split("started job ")[1].split(":")[0]
    job = _get(jid)
    assert job is not None and job.poll()["running"] is True
    # Wait until the worker thread has actually spawned Popen, so the
    # kill goes through the live-proc branch (proc.kill()), not the
    # pre-spawn _kill_requested flag.
    deadline = time.time() + 5
    while job.proc is None and time.time() < deadline:
        time.sleep(0.05)
    assert job.proc is not None and job.proc.poll() is None
    assert run(BgKillTool(), job_id=jid) == f"killed job {jid}"
    state = _wait_done(jid)
    assert state["returncode"] == -9


def test_bg_popen_failure_records_error_125(_workdir, monkeypatch):
    real_popen = subprocess.Popen

    def boom(*args, **kwargs):
        if args and args[0] == ["bash", "-c", "boom-cmd"]:
            raise RuntimeError("injected popen failure")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", boom)
    job = _BgJob("boom-cmd", _workdir, 5)
    job.start()
    job._thread.join(timeout=10)
    state = job.poll()
    assert state["returncode"] == 125
    assert state["error"] == "injected popen failure"
    assert state["running"] is False


# ---- job table: eviction and overflow --------------------------------------

def _fake_job(running):
    job = _BgJob("true", "/tmp", 60)
    job.returncode = None if running else 0
    return job


def test_register_evicts_finished_jobs_at_cap():
    try:
        for _ in range(te.MAX_BG_JOBS):
            _register(_fake_job(running=False))
        assert len(_jobs) == te.MAX_BG_JOBS
        jid = _register(_fake_job(running=True))
        assert len(_jobs) == 1, "finished jobs must be evicted before refusing"
        assert jid in _jobs
    finally:
        with te._jobs_lock:
            _jobs.clear()


def test_register_refuses_when_all_running():
    try:
        for _ in range(te.MAX_BG_JOBS):
            _register(_fake_job(running=True))
        with pytest.raises(RuntimeError, match="too many background jobs running"):
            _register(_fake_job(running=True))
    finally:
        with te._jobs_lock:
            _jobs.clear()


def test_start_bg_blocked_cwd(_workdir):
    assert run(StartBgTool(), command="echo hi", cwd="../x").startswith("Blocked:")


def test_start_bg_reports_table_full(_workdir):
    try:
        for _ in range(te.MAX_BG_JOBS):
            _register(_fake_job(running=True))
        out = run(StartBgTool(), command="echo hi")
        assert out == "too many background jobs running"
    finally:
        with te._jobs_lock:
            _jobs.clear()


# ---- registry --------------------------------------------------------------

def test_exec_tools_registry():
    assert [t.name for t in exec_tools()] == [
        "run_shell", "run_python", "start_bg", "bg_poll", "bg_kill",
    ]
