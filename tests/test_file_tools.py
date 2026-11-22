"""File tools: error branches and the tools test_files.py never touches.

Pins the failure paths of read/write/edit/list/search/mkdir and full
behaviour of append/move/copy/delete/stat plus the file_tools() registry.
Caps are monkeypatched small where the real cap is 5 MB / 200 KB so the
tests stay cheap; no test writes more than a few KB.
"""

import asyncio
import builtins
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import sandbox  # noqa: E402
from app import tools_files as tf  # noqa: E402
from app.tools_files import (  # noqa: E402
    FileAppendTool,
    FileCopyTool,
    FileDeleteTool,
    FileEditTool,
    FileListTool,
    FileMkdirTool,
    FileMoveTool,
    FileReadTool,
    FileSearchTool,
    FileStatTool,
    FileWriteTool,
    file_tools,
)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = str(tmp_path / "work")
    os.makedirs(r)
    monkeypatch.setenv("BRO_WORKDIR", r)
    assert sandbox.workdir() == os.path.abspath(r)
    return r


def run(tool, **kwargs):
    return asyncio.run(tool(**kwargs)).output


def write(root, rel, content, mode="w"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, encoding="utf-8") as f:
        f.write(content)
    return p


@pytest.fixture
def failing_open(monkeypatch):
    """Patch builtins.open to raise OSError for a given mode (or all modes).

    Returns a helper: fail(mode) arms the failure for that open() mode.
    """
    real_open = builtins.open
    modes = set()

    def wrapper(path, mode="r", *args, **kwargs):
        m = mode[0] if isinstance(mode, str) else "r"
        if m in modes:
            raise OSError(f"injected open failure ({mode})")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", wrapper)
    return modes.add


# ---- FileRead --------------------------------------------------------------

def test_read_failed_on_oserror(root, failing_open):
    write(root, "a.txt", "hello")
    failing_open("r")
    assert run(FileReadTool(), path="a.txt").startswith("Read failed:")


def test_read_binary_refused(root):
    write(root, "bin.dat", "x\x00y", mode="w")
    assert "Binary file, not shown" in run(FileReadTool(), path="bin.dat")


def test_read_truncation_note(root, monkeypatch):
    monkeypatch.setattr(tf, "READ_MAX_BYTES", 5)
    write(root, "a.txt", "0123456789")
    out = run(FileReadTool(), path="a.txt")
    assert "01234" in out and "[... truncated ...]" in out


def test_read_max_bytes_floor(root, monkeypatch):
    monkeypatch.setattr(tf, "READ_MAX_BYTES", 5)
    write(root, "a.txt", "0123456789")
    assert "0" in run(FileReadTool(), path="a.txt", max_bytes=0)


# ---- FileWrite -------------------------------------------------------------

def test_write_blocked_over_cap(root, monkeypatch):
    monkeypatch.setattr(tf, "WRITE_MAX_BYTES", 10)
    out = run(FileWriteTool(), path="big.txt", content="x" * 11)
    assert out == "Blocked: content exceeds 10 byte cap"
    assert not os.path.exists(os.path.join(root, "big.txt"))


def test_write_failed_when_parent_is_a_file(root):
    write(root, "blocker", "i am a file")
    out = run(FileWriteTool(), path="blocker/x.txt", content="hi")
    assert out.startswith("Write failed:")


def test_write_creates_nested_parents(root):
    out = run(FileWriteTool(), path="a/b/c.txt", content="deep")
    assert out.startswith("Wrote 4 bytes")
    assert open(os.path.join(root, "a/b/c.txt")).read() == "deep"


# ---- FileEdit --------------------------------------------------------------

def test_edit_blocked_path(root):
    assert run(FileEditTool(), path="../escape.txt", old_text="a", new_text="b").startswith("Blocked:")


def test_edit_not_a_file(root):
    assert run(FileEditTool(), path="missing.txt", old_text="a", new_text="b").startswith("Not a file:")


def test_edit_read_failed(root, failing_open):
    write(root, "a.txt", "hello world")
    failing_open("r")
    assert run(FileEditTool(), path="a.txt", old_text="world", new_text="there").startswith("Read failed:")


def test_edit_result_over_cap(root, monkeypatch):
    monkeypatch.setattr(tf, "WRITE_MAX_BYTES", 10)
    write(root, "a.txt", "hello world")
    out = run(FileEditTool(), path="a.txt", old_text="world", new_text="x" * 20)
    assert out == "Blocked: result would exceed the write size cap"
    assert open(os.path.join(root, "a.txt")).read() == "hello world"


def test_edit_write_failed(root, failing_open):
    write(root, "a.txt", "hello world")
    failing_open("w")
    out = run(FileEditTool(), path="a.txt", old_text="world", new_text="there")
    assert out.startswith("Write failed:")
    assert open(os.path.join(root, "a.txt")).read() == "hello world"


def test_edit_happy_path_and_anchor_errors(root):
    write(root, "a.txt", "one two one")
    assert run(FileEditTool(), path="a.txt", old_text="zzz", new_text="y").startswith("Anchor text not found")
    assert "occurs 2 times" in run(FileEditTool(), path="a.txt", old_text="one", new_text="y")
    assert run(FileEditTool(), path="a.txt", old_text="two", new_text="2").startswith("Edited")
    assert open(os.path.join(root, "a.txt")).read() == "one 2 one"


# ---- FileList --------------------------------------------------------------

def test_list_blocked_and_not_a_directory(root):
    assert run(FileListTool(), path="../x").startswith("Blocked:")
    write(root, "f.txt", "x")
    assert run(FileListTool(), path="f.txt").startswith("Not a directory:")


def test_list_failed_on_oserror(root, monkeypatch):
    def boom(_path, *args, **kwargs):
        raise OSError("injected listdir failure")

    monkeypatch.setattr(os, "listdir", boom)
    assert run(FileListTool(), path=".").startswith("List failed:")


def test_list_hides_and_shows_dotfiles(root):
    write(root, ".hidden", "h")
    write(root, "visible.txt", "v")
    assert ".hidden" not in run(FileListTool(), path=".")
    assert ".hidden" in run(FileListTool(), path=".", show_hidden=True)
    assert "visible.txt" in run(FileListTool(), path=".")


def test_list_dir_suffix_and_dangling_symlink(root):
    os.makedirs(os.path.join(root, "sub"))
    write(root, "sub/f.txt", "x")
    os.symlink("no-such-target", os.path.join(root, "dangling"))
    out = run(FileListTool(), path=".")
    assert "sub/" in out
    assert "dangling (?)" in out


def test_list_truncates_at_cap(root, monkeypatch):
    monkeypatch.setattr(tf, "LIST_MAX_ENTRIES", 2)
    for name in ("a.txt", "b.txt", "c.txt"):
        write(root, name, "x")
    out = run(FileListTool(), path=".")
    assert "... truncated at 2 entries" in out
    assert out.count("bytes)") == 2


def test_list_empty_dir(root):
    os.makedirs(os.path.join(root, "empty"))
    assert run(FileListTool(), path="empty") == "(empty)"


# ---- FileSearch ------------------------------------------------------------

def test_search_blocked_bad_regex_and_not_found(root):
    assert run(FileSearchTool(), pattern="x", path="../y").startswith("Blocked:")
    assert run(FileSearchTool(), pattern="[", path=".").startswith("Bad regex:")
    assert run(FileSearchTool(), pattern="x", path="missing").startswith("Not found:")


def test_search_single_file_target(root):
    write(root, "a.txt", "alpha\nbeta\nalpha beta\n")
    out = run(FileSearchTool(), pattern="beta", path="a.txt")
    assert "a.txt:2: beta" in out and "a.txt:3: alpha beta" in out


def test_search_case_sensitive_flag(root):
    write(root, "a.txt", "Hello\nhello\n")
    assert run(FileSearchTool(), pattern="hello", path=".").count("a.txt:") == 2
    out = run(FileSearchTool(), pattern="hello", path=".", case_sensitive=True)
    assert out.count("a.txt:") == 1 and "a.txt:2:" in out


def test_search_skips_hidden_oversize_and_binary(root, monkeypatch):
    write(root, ".secret", "needle here")
    write(root, "big.txt", "needle " * 100)
    monkeypatch.setattr(tf, "READ_MAX_BYTES", 10)
    with open(os.path.join(root, "bin.dat"), "wb") as f:
        f.write(b"\x00needle")
    write(root, "small.txt", "needle")
    out = run(FileSearchTool(), pattern="needle", path=".")
    assert "small.txt" in out
    assert ".secret" not in out and "big.txt" not in out and "bin.dat" not in out


def test_search_no_matches(root):
    write(root, "a.txt", "nothing here")
    assert run(FileSearchTool(), pattern="zzz", path=".") == "No matches."


def test_search_hits_capped_with_note(root, monkeypatch):
    monkeypatch.setattr(tf, "SEARCH_MAX_MATCHES", 2)
    write(root, "a.txt", "hit\nhit\nhit\nhit\n")
    out = run(FileSearchTool(), pattern="hit", path=".")
    assert out.count("a.txt:") == 2
    assert "... capped at 2 matches" in out


def test_search_stops_walking_past_2000_targets(root, monkeypatch):
    real_walk = os.walk

    def fake_walk(_top):
        yield ("/fake", [], [f"f{i}.txt" for i in range(2001)])
        yield ("/fake2", [], ["extra.txt"])

    monkeypatch.setattr(os, "walk", fake_walk)
    assert run(FileSearchTool(), pattern="x", path=".") == "No matches."
    assert real_walk is not None  # fixture sanity: real walk untouched afterwards


# ---- FileMkdir -------------------------------------------------------------

def test_mkdir_blocked_failed_and_nested(root):
    assert run(FileMkdirTool(), path="../x").startswith("Blocked:")
    write(root, "blocker", "file")
    assert run(FileMkdirTool(), path="blocker/sub").startswith("mkdir failed:")
    assert run(FileMkdirTool(), path="a/b/c").startswith("Created")
    assert os.path.isdir(os.path.join(root, "a/b/c"))


# ---- FileAppend ------------------------------------------------------------

def test_append_blocked_content_and_total_caps(root, monkeypatch):
    assert run(FileAppendTool(), path="../x", content="hi").startswith("Blocked:")
    monkeypatch.setattr(tf, "WRITE_MAX_BYTES", 10)
    assert run(FileAppendTool(), path="a.txt", content="x" * 11).startswith("Blocked: content exceeds")
    write(root, "a.txt", "12345678")
    assert run(FileAppendTool(), path="a.txt", content="xxxxx") == "Blocked: append would exceed the file size cap"
    assert open(os.path.join(root, "a.txt")).read() == "12345678"


def test_append_creates_and_appends(root):
    assert run(FileAppendTool(), path="log.txt", content="one").startswith("Appended 3 bytes")
    assert run(FileAppendTool(), path="log.txt", content=" two").startswith("Appended 4 bytes")
    assert open(os.path.join(root, "log.txt")).read() == "one two"


def test_append_failed_when_parent_is_a_file(root):
    write(root, "blocker", "file")
    assert run(FileAppendTool(), path="blocker/x.txt", content="hi").startswith("Append failed:")


# ---- FileMove --------------------------------------------------------------

def test_move_blocked_not_found_dest_exists(root):
    assert run(FileMoveTool(), src="a.txt", dest="../b.txt").startswith("Blocked:")
    assert run(FileMoveTool(), src="missing.txt", dest="b.txt").startswith("Not found:")
    write(root, "a.txt", "x")
    write(root, "b.txt", "y")
    assert run(FileMoveTool(), src="a.txt", dest="b.txt").startswith("Destination exists")


def test_move_failed_when_dest_parent_is_a_file(root):
    write(root, "a.txt", "x")
    write(root, "blocker", "file")
    assert run(FileMoveTool(), src="a.txt", dest="blocker/b.txt").startswith("Move failed:")
    assert os.path.exists(os.path.join(root, "a.txt"))


def test_move_file_and_directory(root):
    write(root, "a.txt", "x")
    out = run(FileMoveTool(), src="a.txt", dest="sub/b.txt")
    assert out.startswith("Moved")
    assert open(os.path.join(root, "sub/b.txt")).read() == "x"
    assert not os.path.exists(os.path.join(root, "a.txt"))
    os.makedirs(os.path.join(root, "d"))
    assert run(FileMoveTool(), src="d", dest="d2").startswith("Moved")
    assert os.path.isdir(os.path.join(root, "d2"))


# ---- FileCopy --------------------------------------------------------------

def test_copy_blocked_not_a_file_dest_exists_over_cap(root, monkeypatch):
    assert run(FileCopyTool(), src="a.txt", dest="../b.txt").startswith("Blocked:")
    os.makedirs(os.path.join(root, "d"))
    assert run(FileCopyTool(), src="d", dest="b.txt").startswith("Not a file:")
    write(root, "a.txt", "x")
    write(root, "b.txt", "y")
    assert run(FileCopyTool(), src="a.txt", dest="b.txt").startswith("Destination exists")
    monkeypatch.setattr(tf, "WRITE_MAX_BYTES", 2)
    write(root, "big.txt", "12345")
    assert run(FileCopyTool(), src="big.txt", dest="c.txt").startswith("Blocked: file exceeds")


def test_copy_failed_when_dest_parent_is_a_file(root):
    write(root, "a.txt", "x")
    write(root, "blocker", "file")
    assert run(FileCopyTool(), src="a.txt", dest="blocker/b.txt").startswith("Copy failed:")


def test_copy_happy_path_preserves_content(root):
    write(root, "a.txt", "payload")
    assert run(FileCopyTool(), src="a.txt", dest="sub/b.txt").startswith("Copied")
    assert open(os.path.join(root, "sub/b.txt")).read() == "payload"
    assert open(os.path.join(root, "a.txt")).read() == "payload"


# ---- FileDelete ------------------------------------------------------------

def test_delete_blocked_not_a_file_and_success(root):
    assert run(FileDeleteTool(), path="../x").startswith("Blocked:")
    os.makedirs(os.path.join(root, "d"))
    assert run(FileDeleteTool(), path="d").startswith("Not a file:")
    write(root, "a.txt", "x")
    assert run(FileDeleteTool(), path="a.txt").startswith("Deleted")
    assert not os.path.exists(os.path.join(root, "a.txt"))


def test_delete_failed_on_oserror(root, monkeypatch):
    write(root, "a.txt", "x")

    def boom(_path, *args, **kwargs):
        raise OSError("injected unlink failure")

    monkeypatch.setattr(os, "unlink", boom)
    assert run(FileDeleteTool(), path="a.txt").startswith("Delete failed:")
    assert os.path.exists(os.path.join(root, "a.txt"))


# ---- FileStat --------------------------------------------------------------

def test_stat_blocked_not_found_file_and_dir(root):
    assert run(FileStatTool(), path="../x").startswith("Blocked:")
    assert run(FileStatTool(), path="missing").startswith("Not found:")
    write(root, "a.txt", "12345")
    out = run(FileStatTool(), path="a.txt")
    assert "file, 5 bytes, modified" in out
    os.makedirs(os.path.join(root, "d"))
    assert "dir, " in run(FileStatTool(), path="d")


def test_stat_failed_on_oserror(root, monkeypatch):
    target = write(root, "a.txt", "x")
    real_stat = os.stat
    calls = {"n": 0}

    def boom(path, *args, **kwargs):
        # Fail only for the target file, and only on the second stat of it:
        # the first stat is the exists() check in execute, the second is
        # the stat() call itself. sandbox.workdir() stats only the root,
        # so a blanket patch would break the tool before it gets there.
        if os.fspath(path) == target:
            calls["n"] += 1
            if calls["n"] >= 2:
                raise OSError("injected stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", boom)
    assert run(FileStatTool(), path="a.txt").startswith("stat failed:")


# ---- registry --------------------------------------------------------------

def test_file_tools_registry():
    tools = file_tools()
    assert [t.name for t in tools] == [
        "file_read", "file_write", "file_edit", "file_list", "file_search",
        "file_mkdir", "file_append", "file_move", "file_copy",
        "file_delete", "file_stat",
    ]
    assert len({t.name for t in tools}) == 11
