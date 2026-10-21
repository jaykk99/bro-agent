"""Tests for the sandboxed shell tools. No network; quick local commands only."""

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import tools_exec  # noqa: E402
from app.tools_exec import (  # noqa: E402
    BgKillTool,
    BgPollTool,
    RunPythonTool,
    RunShellTool,
    StartBgTool,
    _blocked_reason,
)


@pytest.fixture()
def _workdir(tmp_path, monkeypatch):
    r = str(tmp_path / "work")
    os.makedirs(r)
    monkeypatch.setenv("BRO_WORKDIR", r)
    return r


def run(tool, **kwargs):
    return asyncio.run(tool.run(**kwargs)).output


def test_blocked_patterns():
    assert _blocked_reason("rm -rf /") is not None
    assert _blocked_reason("rm -rf / tmp") is not None
    assert _blocked_reason("rm -rf ~/") is not None
    assert _blocked_reason(":(){ :|:& };:") is not None
    assert _blocked_reason("mkfs.ext4 /dev/sda1") is not None
    assert _blocked_reason("dd if=/dev/zero of=/dev/sda") is not None
    assert _blocked_reason("sudo shutdown now") is not None
    # legit work passes
    assert _blocked_reason("rm -rf ./build") is None
    assert _blocked_reason("npm install && npm test") is None
    assert _blocked_reason("git status") is None
    assert _blocked_reason("python3 train.py --epochs 3") is None


def test_run_shell_ok(_workdir):
    out = run(RunShellTool(), command="echo hello && pwd")
    assert "exit=0" in out
    assert "hello" in out
    assert _workdir in out


def test_run_shell_cwd_sandboxed(_workdir):
    out = run(RunShellTool(), command="pwd", cwd=".")
    assert _workdir in out
    out = run(RunShellTool(), command="pwd", cwd="../../..")
    assert out.startswith("Blocked:")


def test_run_shell_exit_code(_workdir):
    out = run(RunShellTool(), command="exit 3")
    assert "exit=3" in out


def test_run_shell_timeout(_workdir):
    out = run(RunShellTool(), command="sleep 5", timeout_s=1)
    assert "timed out after 1s" in out


def test_run_shell_destructive_blocked(_workdir):
    out = run(RunShellTool(), command="rm -rf /")
    assert out.startswith("Blocked:")


def test_run_shell_access_code_not_in_env(_workdir, monkeypatch):
    monkeypatch.setenv("BRO_ACCESS_CODE", "supersecret123")
    out = run(RunShellTool(), command="env | grep -c ACCESS_CODE || true")
    assert "supersecret123" not in out


def test_run_python(_workdir):
    out = run(RunPythonTool(), code="print(2 + 2)\nprint('done')")
    assert "exit=0" in out
    assert "4" in out and "done" in out


def test_run_python_error(_workdir):
    out = run(RunPythonTool(), code="raise ValueError('boom')")
    assert "ValueError: boom" in out


def test_bg_lifecycle(_workdir):
    out = run(StartBgTool(), command="echo step1; sleep 0.3; echo step2", timeout_s=10)
    jid = out.split("started job")[1].split(":")[0].strip()
    # poll while running, then after exit
    mid = run(BgPollTool(), job_id=jid)
    assert jid in mid
    time.sleep(0.8)
    end = run(BgPollTool(), job_id=jid)
    assert "EXITED" in end
    assert "step2" in end


def test_bg_kill(_workdir):
    out = run(StartBgTool(), command="sleep 30", timeout_s=60)
    jid = out.split("started job")[1].split(":")[0].strip()
    assert "killed job" in run(BgKillTool(), job_id=jid)
    time.sleep(0.5)
    assert "EXITED" in run(BgPollTool(), job_id=jid)


def test_bg_unknown_job(_workdir):
    assert "Unknown job_id" in run(BgPollTool(), job_id="nope")
    assert "Unknown job_id" in run(BgKillTool(), job_id="nope")


def test_bg_destructive_blocked(_workdir):
    out = run(StartBgTool(), command="mkfs.ext4 /dev/sda1")
    assert out.startswith("Blocked:")
