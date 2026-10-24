"""Tests for the git tools. Uses throwaway repos under a temp workdir."""

import asyncio
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import tools_git  # noqa: E402
from app.tools_git import (  # noqa: E402
    GitAddTool,
    GitBranchTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitPullTool,
    GitPushTool,
    GitStatusTool,
    _refused_reason,
)


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("BRO_WORKDIR", str(work))
    r = work / "demo"
    r.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=r, check=True)
    (r / "a.txt").write_text("one\n")
    return r


def run(tool, **kwargs):
    # The real UniHarness BaseAgentTool has no .run(); tools are invoked
    # through __call__, which validates kwargs against args_schema.
    return asyncio.run(tool(**kwargs)).output


def test_refused_patterns():
    assert _refused_reason(["push", "--force"]) is not None
    assert _refused_reason(["push", "-f"]) is not None
    assert _refused_reason(["reset", "--hard"]) is not None
    assert _refused_reason(["clean", "-fd"]) is not None
    assert _refused_reason(["filter-branch"]) is not None
    assert _refused_reason(["checkout", "--", "a.txt"]) is not None
    assert _refused_reason(["status"]) is None
    assert _refused_reason(["push", "origin", "main"]) is None
    assert _refused_reason(["pull", "--ff-only", "origin"]) is None


def test_not_a_repo(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("BRO_WORKDIR", str(work))
    assert "Not a git repo" in run(GitStatusTool(), repo=".")


def test_repo_outside_workdir_blocked(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("BRO_WORKDIR", str(work))
    assert run(GitStatusTool(), repo="../..").startswith("Blocked:")


def test_status_log_diff_flow(repo):
    out = run(GitStatusTool(), repo="demo")
    assert "a.txt" in out  # untracked shows up
    assert "staged" in run(GitAddTool(), repo="demo")
    assert "nothing to commit" not in run(GitCommitTool(), repo="demo", message="first")
    out = run(GitLogTool(), repo="demo", n=5)
    assert "first" in out
    (repo / "a.txt").write_text("one\ntwo\n")
    out = run(GitDiffTool(), repo="demo")
    assert "+two" in out
    out = run(GitStatusTool(), repo="demo")
    assert " M a.txt" in out or "M a.txt" in out


def test_commit_empty_message(repo):
    assert "Empty commit message" in run(GitCommitTool(), repo="demo", message="  ")


def test_branch_create_and_list(repo):
    run(GitAddTool(), repo="demo")
    run(GitCommitTool(), repo="demo", message="base")
    out = run(GitBranchTool(), repo="demo", name="feature-x")
    assert "exit=0" in out
    out = run(GitBranchTool(), repo="demo")
    assert "feature-x" in out


def test_branch_bad_name(repo):
    assert "Bad branch name" in run(GitBranchTool(), repo="demo", name="bad;name")


def test_push_force_refused_even_via_tool(repo):
    # Sneaky flag smuggled inside the branch/remote args must still be refused.
    out = run(GitPushTool(), repo="demo", branch="main --force")
    assert "Blocked:" in out
    out = run(GitPushTool(), repo="demo", remote="origin --force")
    assert "Blocked:" in out


def test_pull_ff_only_flag_present(repo, monkeypatch):
    # Point at a repo with no remote: pull fails, but must have used --ff-only.
    calls = []
    orig = tools_git._run_git

    def spy(repo_p, args, timeout_s=60):
        calls.append(args)
        return orig(repo_p, args, timeout_s)

    monkeypatch.setattr(tools_git, "_run_git", spy)
    run(GitPullTool(), repo="demo")
    assert calls and "--ff-only" in calls[0]
