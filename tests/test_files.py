"""Tests for the workdir sandbox and file tools. No network."""

import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import sandbox  # noqa: E402
from app.sandbox import SandboxViolation, display_path, resolve  # noqa: E402
from app.tools_files import (  # noqa: E402
    FileEditTool,
    FileListTool,
    FileMkdirTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = str(tmp_path / "work")
    os.makedirs(r)
    monkeypatch.setenv("BRO_WORKDIR", r)
    assert sandbox.workdir() == os.path.abspath(r)
    return r


def run(tool, **kwargs):
    # The real UniHarness BaseAgentTool has no .run(); tools are invoked
    # through __call__, which validates kwargs against args_schema.
    return asyncio.run(tool(**kwargs)).output


def test_resolve_inside(root):
    assert resolve(root, "a/b.txt") == os.path.join(root, "a", "b.txt")
    assert resolve(root, ".") == root


def test_resolve_dotdot_escape_blocked(root):
    with pytest.raises(SandboxViolation):
        resolve(root, "../../etc/passwd")


def test_resolve_absolute_outside_blocked(root):
    with pytest.raises(SandboxViolation):
        resolve(root, "/etc/passwd")


def test_resolve_symlink_escape_blocked(root):
    outside = os.path.join(os.path.dirname(root), "outside")
    os.makedirs(outside)
    os.symlink(outside, os.path.join(root, "link"))
    with pytest.raises(SandboxViolation):
        resolve(root, "link/secret.txt")


def test_resolve_tilde_blocked(root):
    with pytest.raises(SandboxViolation):
        resolve(root, "~/x")


def test_display_path(root):
    assert display_path(root, root) == "."
    assert display_path(root, os.path.join(root, "a")) == "a"


def test_write_read_roundtrip(root):
    out = run(FileWriteTool(), path="notes/todo.txt", content="hello\nworld")
    assert "Wrote" in out
    out = run(FileReadTool(), path="notes/todo.txt")
    assert "hello\nworld" in out


def test_read_missing(root):
    assert "Not a file" in run(FileReadTool(), path="nope.txt")


def test_read_binary_refused(root):
    with open(os.path.join(root, "b.bin"), "wb") as f:
        f.write(b"\x00\x01\x02binary")
    assert "Binary" in run(FileReadTool(), path="b.bin")


def test_write_outside_blocked(root):
    out = run(FileWriteTool(), path="../evil.txt", content="x")
    assert out.startswith("Blocked:")


def test_edit_exact_anchor(root):
    run(FileWriteTool(), path="f.txt", content="alpha\nbeta\ngamma\n")
    out = run(FileEditTool(), path="f.txt", old_text="beta\n", new_text="BETA\n")
    assert "Edited" in out
    assert "BETA" in run(FileReadTool(), path="f.txt")


def test_edit_missing_anchor(root):
    run(FileWriteTool(), path="f.txt", content="alpha\n")
    assert "not found" in run(FileEditTool(), path="f.txt", old_text="zzz", new_text="q")


def test_edit_ambiguous_anchor(root):
    run(FileWriteTool(), path="f.txt", content="x\nx\nx\n")
    assert "occurs 3 times" in run(FileEditTool(), path="f.txt", old_text="x\n", new_text="y\n")


def test_list_dir(root):
    run(FileWriteTool(), path="a.txt", content="1")
    run(FileMkdirTool(), path="sub")
    out = run(FileListTool(), path=".")
    assert "a.txt" in out and "sub/" in out


def test_search(root):
    run(FileWriteTool(), path="one.txt", content="the quick brown fox")
    run(FileWriteTool(), path="two.txt", content="lazy dog")
    out = run(FileSearchTool(), pattern="quick|lazy")
    assert "one.txt:1" in out and "two.txt:1" in out
    assert "No matches" in run(FileSearchTool(), pattern="zebra")
    assert "Bad regex" in run(FileSearchTool(), pattern="([")


def test_append_move_copy_delete_stat(root):
    from app.tools_files import (
        FileAppendTool,
        FileCopyTool,
        FileDeleteTool,
        FileMoveTool,
        FileStatTool,
    )

    run(FileWriteTool(), path="a.txt", content="one\n")
    assert "Appended" in run(FileAppendTool(), path="a.txt", content="two\n")
    assert "two" in run(FileReadTool(), path="a.txt")
    assert "Moved" in run(FileMoveTool(), src="a.txt", dest="sub/b.txt")
    assert "Not a file" in run(FileReadTool(), path="a.txt")
    assert "Copied" in run(FileCopyTool(), src="sub/b.txt", dest="c.txt")
    assert "one" in run(FileReadTool(), path="c.txt")
    out = run(FileStatTool(), path="c.txt")
    assert "file" in out and "bytes" in out
    assert "Destination exists" in run(FileCopyTool(), src="sub/b.txt", dest="c.txt")
    assert "Deleted" in run(FileDeleteTool(), path="c.txt")
    assert "Not a file" in run(FileDeleteTool(), path="c.txt")
    # directories are never deleted by file_delete
    run(FileMkdirTool(), path="d")
    assert "Not a file" in run(FileDeleteTool(), path="d")
    # sandbox still applies
    assert run(FileDeleteTool(), path="../x").startswith("Blocked:")
    assert run(FileMoveTool(), src="sub/b.txt", dest="../../evil").startswith("Blocked:")
