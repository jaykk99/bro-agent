"""Git tools for the agent — repo workflows inside the workdir sandbox.

Jay is a solo dev and the agent does real repo work, so it needs git:
status, diff, log, add, commit, branch, push, pull. Every command runs
with cwd jailed to BRO_WORKDIR (a repo must live under the workdir —
~/bro-work/<repo>).

Safety rules, enforced in code, not just prompting:
  * `git push --force` / `-f` is refused, always. History rewrites need Jay.
  * `git reset --hard`, `git clean -fd`, `git checkout -- .` style
    destructive commands are refused — propose, don't destroy.
  * `filter-branch`, `filter-repo`, `rebase` onto rewritten history are refused.
  * Commits always go through; pushing is allowed (Render deploys on push
    for his sites — that's the normal workflow).
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from .sandbox import SandboxViolation, display_path, resolve, workdir
from .tool_base import BaseAgentTool, ToolResult
from .tools_exec import _safe_env

MAX_GIT_OUTPUT = 8000

# Refused outright — destructive or history-rewriting.
_REFUSED = [
    (re.compile(r"--force\b|\s-f(\s|$)"), "force-push / force flag"),
    (re.compile(r"\breset\s+--hard\b"), "hard reset"),
    (re.compile(r"\bclean\s+-[a-z]*f"), "git clean -f"),
    (re.compile(r"\bfilter-branch\b|\bfilter-repo\b"), "history filter"),
    (re.compile(r"\bpush\b.*\b--delete\b"), "remote branch delete"),
]


def _refused_reason(args: list[str]) -> str | None:
    joined = " ".join(args)
    for rx, label in _REFUSED:
        if rx.search(joined):
            return f"Blocked: destructive git operation ({label}) — ask Jay instead"
    # `git checkout -- <paths>` / `git restore` discarding changes: refuse
    # only when it would discard work without a commit-ish target.
    if args[:1] == ["checkout"] and "--" in args:
        return "Blocked: 'git checkout -- <paths>' discards uncommitted work — ask Jay instead"
    return None


def _run_git(repo: str, args: list[str], timeout_s: int = 60) -> tuple[int, str]:
    refused = _refused_reason(args)
    if refused:
        return 126, refused
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            env=_safe_env(),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_GIT_OUTPUT:
            out = out[:MAX_GIT_OUTPUT] + "\n[... capped ...]"
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, "git timed out"
    except FileNotFoundError:
        return 127, "git is not installed"


def _repo_dir(path: str) -> tuple[str | None, str]:
    """Resolve the repo dir inside the workdir. Returns (abs_path, error)."""
    root = workdir()
    try:
        p = resolve(root, path or ".")
    except SandboxViolation as e:
        return None, f"Blocked: {e}"
    if not os.path.isdir(os.path.join(p, ".git")):
        return None, f"Not a git repo: {display_path(root, p)}"
    return p, ""


class GitRepoParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")


class GitStatusTool(BaseAgentTool[GitRepoParams]):
    name: str = "git_status"
    description: str = "git status (short) + current branch for a repo in the workdir."
    args_schema = GitRepoParams

    async def execute(self, params: GitRepoParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        code, out = _run_git(repo, ["status", "--short", "--branch"])
        return ToolResult(output=f"exit={code}\n{out}" if out else f"exit={code}\n(clean)")


class GitDiffParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")
    staged: bool = Field(default=False, description="Show staged diff instead of unstaged")
    path: str = Field(default="", description="Optional single path to diff")


class GitDiffTool(BaseAgentTool[GitDiffParams]):
    name: str = "git_diff"
    description: str = "Show the diff of unstaged (or staged) changes."
    args_schema = GitDiffParams

    async def execute(self, params: GitDiffParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        args = ["diff"]
        if params.staged:
            args.append("--cached")
        args += ["--", params.path] if params.path else []
        code, out = _run_git(repo, args)
        return ToolResult(output=f"exit={code}\n{out}" if out else f"exit={code}\n(no diff)")


class GitLogParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")
    n: int = Field(default=10, description="Number of commits to show")


class GitLogTool(BaseAgentTool[GitLogParams]):
    name: str = "git_log"
    description: str = "Recent commits, one line each with hash, date, subject."
    args_schema = GitLogParams

    async def execute(self, params: GitLogParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        n = max(1, min(params.n, 50))
        code, out = _run_git(repo, ["log", f"-{n}", "--pretty=format:%h %ad %s", "--date=short"])
        return ToolResult(output=f"exit={code}\n{out}" if out else f"exit={code}\n(no commits)")


class GitAddParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")
    paths: list[str] = Field(default_factory=list, description="Paths to stage (relative to repo); empty = all")


class GitAddTool(BaseAgentTool[GitAddParams]):
    name: str = "git_add"
    description: str = "Stage files for commit."
    args_schema = GitAddParams

    async def execute(self, params: GitAddParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        paths = params.paths or ["-A"]
        args = ["add", "-A"] if paths == ["-A"] else ["add", "--", *paths]
        code, out = _run_git(repo, args)
        if code != 0:
            return ToolResult(output=f"exit={code}\n{out}")
        _, status = _run_git(repo, ["status", "--short"])
        return ToolResult(output=f"staged.\n{status}" if status else "staged. (nothing to commit)")


class GitCommitParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")
    message: str = Field(description="Commit message")


class GitCommitTool(BaseAgentTool[GitCommitParams]):
    name: str = "git_commit"
    description: str = "Commit staged changes. Never amends, never force-pushes."
    args_schema = GitCommitParams

    async def execute(self, params: GitCommitParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        msg = (params.message or "").strip()
        if not msg:
            return ToolResult(output="Empty commit message — say what changed.")
        # Never allow amend / --no-verify sneaks through the message.
        code, out = _run_git(repo, ["commit", "-m", msg])
        return ToolResult(output=f"exit={code}\n{out}")


class GitBranchParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")
    name: str = Field(default="", description="Create and switch to this branch (empty = list branches)")
    delete: str = Field(default="", description="Delete this local branch (never the current one)")


class GitBranchTool(BaseAgentTool[GitBranchParams]):
    name: str = "git_branch"
    description: str = "List branches, or create-and-switch to a new branch, or delete a merged local branch."
    args_schema = GitBranchParams

    async def execute(self, params: GitBranchParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        if params.delete:
            code, out = _run_git(repo, ["branch", "-d", params.delete])
            return ToolResult(output=f"exit={code}\n{out}")
        if params.name:
            if not re.fullmatch(r"[A-Za-z0-9._/-]+", params.name):
                return ToolResult(output="Bad branch name (letters, digits, . _ / - only).")
            code, out = _run_git(repo, ["checkout", "-b", params.name])
            return ToolResult(output=f"exit={code}\n{out}")
        code, out = _run_git(repo, ["branch", "--list"])
        return ToolResult(output=f"exit={code}\n{out}")


class GitPushParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")
    remote: str = Field(default="origin", description="Remote name")
    branch: str = Field(default="", description="Branch to push (empty = current)")


class GitPushTool(BaseAgentTool[GitPushParams]):
    name: str = "git_push"
    description: str = "Push the current (or named) branch. Force-push is refused, always."
    args_schema = GitPushParams

    async def execute(self, params: GitPushParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        args = ["push", params.remote]
        if params.branch:
            args.append(params.branch)
        code, out = _run_git(repo, args, timeout_s=120)
        return ToolResult(output=f"exit={code}\n{out}")


class GitPullParams(BaseModel):
    repo: str = Field(default=".", description="Repo dir relative to the workdir")
    remote: str = Field(default="origin", description="Remote name")


class GitPullTool(BaseAgentTool[GitPullParams]):
    name: str = "git_pull"
    description: str = "Pull latest (fast-forward only — refuses to merge/rebase)."
    args_schema = GitPullParams

    async def execute(self, params: GitPullParams) -> ToolResult:
        repo, err = _repo_dir(params.repo)
        if err:
            return ToolResult(output=err)
        code, out = _run_git(repo, ["pull", "--ff-only", params.remote], timeout_s=120)
        return ToolResult(output=f"exit={code}\n{out}")


def git_tools() -> list[Any]:
    return [
        GitStatusTool(),
        GitDiffTool(),
        GitLogTool(),
        GitAddTool(),
        GitCommitTool(),
        GitBranchTool(),
        GitPushTool(),
        GitPullTool(),
    ]
