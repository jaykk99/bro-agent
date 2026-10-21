"""Shell execution tools for the agent — jailed to BRO_WORKDIR.

run_shell / run_python execute inside the workdir sandbox with a hard
timeout, capped output, and a small blocklist of catastrophic patterns
(rm -rf /, fork bombs, mkfs, raw disk writes, power ops). Legit work —
builds, scripts, git, tests — passes through untouched.

start_bg / bg_poll / bg_kill give the agent long-running jobs: start a
process in the background, poll its output incrementally, kill it when
done. Jobs live in-process (Render free tier is single-process anyway).
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from .sandbox import SandboxViolation, display_path, resolve, workdir
from .tool_base import BaseAgentTool, ToolResult

DEFAULT_TIMEOUT_S = 60
MAX_OUTPUT_CHARS = 12000

# Catastrophic-only blocklist. This is not a nanny filter: builds, installs,
# git, curl, and normal scripting all pass. Only patterns that destroy the
# host, the disk, or fork-bomb the box are refused.
_BLOCKED_PATTERNS = [
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\s+/(?:\s|$)"), "rm -rf /"),
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\s+~(?:\s|/|$)"), "rm -rf ~"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"), "fork bomb"),
    (re.compile(r"\bmkfs(\.|s)?\b"), "mkfs"),
    (re.compile(r"\bdd\s+.*\bof=/dev/"), "dd to a device"),
    (re.compile(r">\s*/dev/sd[a-z]"), "raw disk write"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "power operation"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/\b"), "chmod 777 /"),
]


def _blocked_reason(cmd: str) -> str | None:
    for rx, label in _BLOCKED_PATTERNS:
        if rx.search(cmd):
            return f"Blocked: destructive pattern ({label})"
    return None


def _run_sync(cmd: str, timeout_s: int, cwd: str) -> tuple[int, str]:
    """Run cmd via bash -c. Returns (returncode, combined output capped)."""
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            cwd=cwd,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            env=_safe_env(),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n[... output capped at {MAX_OUTPUT_CHARS} chars ...]"
        return proc.returncode, out
    except subprocess.TimeoutExpired as e:
        partial = ((e.stdout or "") + (e.stderr or ""))[:MAX_OUTPUT_CHARS]
        return 124, f"[timed out after {timeout_s}s]\n{partial}"


def _safe_env() -> dict[str, str]:
    """Subprocess env: inherit, but never hand the access code to a shell."""
    env = dict(os.environ)
    env.pop("BRO_ACCESS_CODE", None)
    return env


class RunShellParams(BaseModel):
    command: str = Field(description="Shell command to run (bash -c)")
    timeout_s: int = Field(default=DEFAULT_TIMEOUT_S, description="Kill after this many seconds (1-300)")
    cwd: str = Field(default=".", description="Working directory, relative to the workdir")


class RunShellTool(BaseAgentTool[RunShellParams]):
    name: str = "run_shell"
    description: str = (
        "Run a shell command in the workdir sandbox. Returns exit code and "
        "output (capped). Times out instead of hanging."
    )
    args_schema = RunShellParams

    async def execute(self, params: RunShellParams) -> ToolResult:
        root = workdir()
        blocked = _blocked_reason(params.command)
        if blocked:
            return ToolResult(output=blocked)
        try:
            cwd = resolve(root, params.cwd)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        timeout = max(1, min(params.timeout_s, 300))
        code, out = await _run_in_thread(params.command, timeout, cwd)
        return ToolResult(output=f"exit={code} cwd={display_path(root, cwd)}\n{out}".rstrip())


async def _run_in_thread(cmd: str, timeout_s: int, cwd: str) -> tuple[int, str]:
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_sync, cmd, timeout_s, cwd)


class RunPythonParams(BaseModel):
    code: str = Field(description="Python code to execute")
    timeout_s: int = Field(default=DEFAULT_TIMEOUT_S, description="Kill after this many seconds (1-300)")


class RunPythonTool(BaseAgentTool[RunPythonParams]):
    name: str = "run_python"
    description: str = (
        "Execute Python code in the workdir sandbox (subprocess, stdout "
        "captured, capped, timed out). Imports are unrestricted — this is "
        "Jay's own machine for his own work."
    )
    args_schema = RunPythonParams

    async def execute(self, params: RunPythonParams) -> ToolResult:
        root = workdir()
        blocked = _blocked_reason(params.code)
        if blocked:
            return ToolResult(output=blocked)
        timeout = max(1, min(params.timeout_s, 300))
        # Write to a temp file under the workdir instead of `python -c`,
        # so tracebacks show real line numbers and big scripts work.
        script = os.path.join(root, f".bro-tmp-{uuid.uuid4().hex[:8]}.py")
        try:
            with open(script, "w", encoding="utf-8") as f:
                f.write(params.code)
            code, out = await _run_in_thread(f"python3 {script!r}", timeout, root)
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass
        return ToolResult(output=f"exit={code}\n{out}".rstrip())


# ---- Background jobs ----


class _BgJob:
    def __init__(self, cmd: str, cwd: str, timeout_s: int):
        self.id = uuid.uuid4().hex[:8]
        self.cmd = cmd
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.started_at = time.time()
        self.proc: subprocess.Popen | None = None
        self.output = ""
        self.returncode: int | None = None
        self.error = ""
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            self.proc = subprocess.Popen(
                ["bash", "-c", self.cmd],
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=_safe_env(),
            )
            try:
                out, _ = self.proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                out, _ = self.proc.communicate()
                out = (out or "") + f"\n[killed after {self.timeout_s}s timeout]"
            with self._lock:
                self.output = (out or "")[:MAX_OUTPUT_CHARS]
                self.returncode = self.proc.returncode
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.error = str(e)
                self.returncode = 125

    def poll(self, max_chars: int = 4000) -> dict:
        with self._lock:
            running = self.returncode is None and not self.error
            return {
                "id": self.id,
                "running": running,
                "returncode": self.returncode,
                "error": self.error,
                "elapsed_s": round(time.time() - self.started_at, 1),
                "output_tail": self.output[-max_chars:],
            }

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


_jobs: dict[str, _BgJob] = {}
_jobs_lock = threading.Lock()
MAX_BG_JOBS = 20


def _register(job: _BgJob) -> str:
    with _jobs_lock:
        if len(_jobs) >= MAX_BG_JOBS:
            # evict finished jobs first
            for jid in [k for k, j in _jobs.items() if not j.poll()["running"]]:
                del _jobs[jid]
        if len(_jobs) >= MAX_BG_JOBS:
            raise RuntimeError("too many background jobs running")
        _jobs[job.id] = job
    return job.id


def _get(job_id: str) -> _BgJob | None:
    with _jobs_lock:
        return _jobs.get(job_id)


class StartBgParams(BaseModel):
    command: str = Field(description="Shell command to run in the background")
    timeout_s: int = Field(default=600, description="Kill after this many seconds (1-3600)")
    cwd: str = Field(default=".", description="Working directory, relative to the workdir")


class StartBgTool(BaseAgentTool[StartBgParams]):
    name: str = "start_bg"
    description: str = (
        "Start a long-running shell command in the background. Returns a "
        "job_id — poll it with bg_poll, stop it with bg_kill."
    )
    args_schema = StartBgParams

    async def execute(self, params: StartBgParams) -> ToolResult:
        root = workdir()
        blocked = _blocked_reason(params.command)
        if blocked:
            return ToolResult(output=blocked)
        try:
            cwd = resolve(root, params.cwd)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        timeout = max(1, min(params.timeout_s, 3600))
        job = _BgJob(params.command, cwd, timeout)
        try:
            jid = _register(job)
        except RuntimeError as e:
            return ToolResult(output=str(e))
        job.start()
        return ToolResult(output=f"started job {jid}: {params.command[:120]}")


class BgPollParams(BaseModel):
    job_id: str = Field(description="Job id from start_bg")


class BgPollTool(BaseAgentTool[BgPollParams]):
    name: str = "bg_poll"
    description: str = "Check a background job: running state, elapsed time, output tail."
    args_schema = BgPollParams

    async def execute(self, params: BgPollParams) -> ToolResult:
        job = _get(params.job_id)
        if not job:
            return ToolResult(output=f"Unknown job_id: {params.job_id}")
        s = job.poll()
        status = "RUNNING" if s["running"] else f"EXITED code={s['returncode']}"
        return ToolResult(
            output=f"job {s['id']}: {status} elapsed={s['elapsed_s']}s\n{ s['output_tail']}".rstrip()
        )


class BgKillParams(BaseModel):
    job_id: str = Field(description="Job id from start_bg")


class BgKillTool(BaseAgentTool[BgKillParams]):
    name: str = "bg_kill"
    description: str = "Kill a background job."
    args_schema = BgKillParams

    async def execute(self, params: BgKillParams) -> ToolResult:
        job = _get(params.job_id)
        if not job:
            return ToolResult(output=f"Unknown job_id: {params.job_id}")
        job.kill()
        return ToolResult(output=f"killed job {params.job_id}")


def exec_tools() -> list[Any]:
    return [RunShellTool(), RunPythonTool(), StartBgTool(), BgPollTool(), BgKillTool()]
