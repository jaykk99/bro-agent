"""Work-files browser — let Jay retrieve what the agent wrote.

The agent writes files into BRO_WORKDIR (reports, scripts, downloads),
but until now the only way to get one out was the terminal: the API
listed `uploads/` and nothing could be downloaded at all. This module
is the read side of the workdir: list a directory, open a file for
download, delete a file.

Security mirrors the train-adapter routes (see app/main.py): paths are
jailed by `sandbox.resolve` (so `..` escapes and symlinks pointing
outside the workdir are refused), and every open/unlink goes through a
file descriptor pinned to the parent directory (O_DIRECTORY | O_NOFOLLOW
+ openat/unlinkat with O_NOFOLLOW), so a symlink swapped in after the
check cannot redirect the operation outside the tree. Directories are
never served or deleted through these helpers — listing is the only
way to look inside one.
"""

from __future__ import annotations

import os
import stat

from .sandbox import resolve

# Bound a single listing: a workdir can hold thousands of agent outputs
# and the ops UI renders one table row per entry.
MAX_LIST_ENTRIES = 500


def _open_parent_dir(root: str, abs_path: str) -> tuple[int, str]:
    """Open the parent dir of `abs_path` pinned (no symlink follow) and
    return (dir_fd, basename). The caller must os.close(dir_fd)."""
    parent = os.path.dirname(abs_path)
    name = os.path.basename(abs_path)
    dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    return dir_fd, name


def list_dir(root: str, rel: str = "") -> list[dict]:
    """List one workdir directory (non-recursive).

    `rel` is "" for the workdir root. Returns entries sorted
    directories-first, then by name: {name, path, is_dir, bytes, mtime}.
    `bytes` is None for directories. Raises FileNotFoundError for a
    missing path and NotADirectoryError when `rel` is a file."""
    target = resolve(root, rel or ".")
    if not os.path.isdir(target):
        if os.path.exists(target):
            raise NotADirectoryError(f"not a directory: {rel!r}")
        raise FileNotFoundError(f"no such directory: {rel!r}")
    entries: list[dict] = []
    for name in sorted(os.listdir(target)):
        path = os.path.join(target, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        is_dir = stat.S_ISDIR(st.st_mode)
        rel_path = os.path.relpath(path, root).replace(os.sep, "/")
        entries.append(
            {
                "name": name,
                "path": rel_path,
                "is_dir": is_dir,
                "bytes": None if is_dir else st.st_size,
                "mtime": st.st_mtime,
            }
        )
        if len(entries) >= MAX_LIST_ENTRIES:
            break
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    return entries


def open_work_file(root: str, rel: str) -> tuple[int, int, str]:
    """Open a workdir file for download, fd-pinned.

    Returns (fd, size, name); the caller owns the fd and must close it.
    Raises FileNotFoundError (missing), IsADirectoryError (a directory),
    ValueError (not a regular file — symlink, socket, device, ...).
    The open goes through the pinned parent dir fd, so a symlink
    swapped in for the final component after resolve() fails with
    ELOOP instead of being followed."""
    target = resolve(root, rel)
    dir_fd, name = _open_parent_dir(root, target)
    try:
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=dir_fd,
        )
    except OSError:
        os.close(dir_fd)
        raise
    os.close(dir_fd)
    try:
        st = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise
    if stat.S_ISDIR(st.st_mode):
        os.close(fd)
        raise IsADirectoryError(f"is a directory: {rel!r}")
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise ValueError(f"not a regular file: {rel!r}")
    return fd, st.st_size, name


def delete_work_file(root: str, rel: str) -> str:
    """Delete one regular workdir file, fd-pinned.

    Returns the file's basename. Raises FileNotFoundError (missing),
    IsADirectoryError (a directory), ValueError (not a regular file).
    The unlink goes through the pinned parent dir fd, so a symlink
    swapped in for the final component is unlinked as an entry, never
    followed to its target."""
    target = resolve(root, rel)
    dir_fd, name = _open_parent_dir(root, target)
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if stat.S_ISDIR(st.st_mode):
            raise IsADirectoryError(f"is a directory: {rel!r}")
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"not a regular file: {rel!r}")
        os.unlink(name, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    return name
