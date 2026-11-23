"""Pin the remaining untested paths in app/tools_git.py.

Coverage gaps left after test_git.py: the _run_git output cap,
timeout, and git-missing branches, each tool's repo-error return,
the staged diff flag, the add-failure branch, branch deletion, and
the git_tools() registry.
"""

import asyncio
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.tools_git as tools_git  # noqa: E402
from app.tools_git import (  # noqa: E402
    GitAddTool,
    GitBranchTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitPullTool,
    GitPushTool,
    GitStatusTool,
    _run_git,
    git_tools,
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


@pytest.fixture()
def plain_dir(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("BRO_WORKDIR", str(work))
    d = work / "plain"
    d.mkdir()
    return d


def run(tool, **kwargs):
    return asyncio.run(tool(**kwargs)).output


# ---- _run_git branches ----


def test_run_git_caps_huge_output(repo):
    for i in range(150):
        (repo / f"file-{i:03d}-{'x' * 40}.txt").write_text("x\n")
    code, out = _run_git(str(repo), ["status", "--short"])
    assert code == 0
    assert "[... capped ...]" in out
    assert len(out) <= tools_git.MAX_GIT_OUTPUT + 30


def test_run_git_timeout(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=60)

    monkeypatch.setattr(subprocess, "run", boom)
    assert _run_git("/tmp", ["status"]) == (124, "git timed out")


def test_run_git_missing_git_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _run_git("/tmp", ["status"]) == (127, "git is not installed")


# ---- every tool's repo-error return ----


def test_each_tool_reports_non_repo(plain_dir):
    for tool, kwargs in [
        (GitStatusTool(), {}),
        (GitDiffTool(), {}),
        (GitLogTool(), {}),
        (GitAddTool(), {}),
        (GitCommitTool(), {"message": "x"}),
        (GitBranchTool(), {}),
        (GitPushTool(), {}),
        (GitPullTool(), {}),
    ]:
        assert "Not a git repo" in run(tool, repo="plain", **kwargs)


# ---- staged diff + path filter ----


def test_diff_staged_and_path_filter(repo):
    run(GitAddTool(), repo="demo")
    run(GitCommitTool(), repo="demo", message="base")
    (repo / "a.txt").write_text("one\ntwo\n")
    (repo / "b.txt").write_text("b\n")
    assert "two" not in run(GitDiffTool(), repo="demo", staged=True)  # nothing staged
    assert "(no diff)" in run(GitDiffTool(), repo="demo", staged=True)
    run(GitAddTool(), repo="demo", paths=["a.txt"])
    staged = run(GitDiffTool(), repo="demo", staged=True)
    assert "+two" in staged
    assert "b.txt" not in staged  # b.txt is unstaged, not in the staged diff
    by_path = run(GitDiffTool(), repo="demo", path="b.txt")
    assert "b.txt" not in by_path  # unstaged b.txt is untracked: no diff entry
    unstaged = run(GitDiffTool(), repo="demo")
    assert "(no diff)" in unstaged  # a.txt is staged now; b.txt untracked


# ---- add failure / branch delete ----


def test_add_nonexistent_path_reports_exit_code(repo):
    out = run(GitAddTool(), repo="demo", paths=["nope.txt"])
    assert out.startswith("exit=128")


def test_branch_delete_round_trip(repo):
    run(GitAddTool(), repo="demo")
    run(GitCommitTool(), repo="demo", message="base")
    run(GitBranchTool(), repo="demo", name="feature-x")
    # feature-x is checked out now; deleting the current branch fails,
    # so switch back first.
    subprocess.run(["git", "checkout", "-q", "-"], cwd=repo, check=True)
    out = run(GitBranchTool(), repo="demo", delete="feature-x")
    assert "exit=0" in out
    assert "feature-x" not in run(GitBranchTool(), repo="demo")
    out = run(GitBranchTool(), repo="demo", delete="no-such-branch")
    assert "exit=" in out and "error" in out.lower()


def test_push_and_pull_without_remote_report_exit_code(repo):
    run(GitAddTool(), repo="demo")
    run(GitCommitTool(), repo="demo", message="base")
    assert "exit=128" in run(GitPushTool(), repo="demo")
    assert "exit=" in run(GitPullTool(), repo="demo")


# ---- registry ----


def test_git_tools_registry():
    tools = git_tools()
    assert [t.name for t in tools] == [
        "git_status",
        "git_diff",
        "git_log",
        "git_add",
        "git_commit",
        "git_branch",
        "git_push",
        "git_pull",
    ]
