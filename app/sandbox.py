"""Workdir sandbox — every agent file operation is jailed here.

All paths the agent touches are resolved against BRO_WORKDIR (default
~/bro-work) with realpath, so `../..` escapes and symlink tricks can never
reach outside the sandbox. Anything outside raises SandboxViolation.
"""

from __future__ import annotations

import os


class SandboxViolation(Exception):
    """Raised when a path escapes the agent's workdir sandbox."""


def workdir() -> str:
    root = os.environ.get("BRO_WORKDIR", os.path.expanduser("~/bro-work"))
    root = os.path.abspath(os.path.expanduser(root))
    os.makedirs(root, exist_ok=True)
    return root


def resolve(root: str, path: str) -> str:
    """Resolve `path` (absolute or relative) inside `root`.

    Returns the absolute real path. Raises SandboxViolation if it escapes.
    """
    root = os.path.abspath(root)
    p = (path or "").strip()
    if not p:
        raise SandboxViolation("empty path")
    if p.startswith("~"):
        raise SandboxViolation("home-directory paths are not allowed")
    abs_p = p if os.path.isabs(p) else os.path.join(root, p)
    # normpath first so plain ".." escapes fail fast; realpath then
    # defeats symlink escapes (a symlink inside root pointing outside).
    normed = os.path.normpath(abs_p)
    if normed != root and not normed.startswith(root + os.sep):
        raise SandboxViolation(f"path escapes workdir: {path!r}")
    real = os.path.realpath(normed)
    if real != root and not real.startswith(root + os.sep):
        raise SandboxViolation(f"path escapes workdir via symlink: {path!r}")
    return real


def display_path(root: str, abs_path: str) -> str:
    """Shorten an absolute sandbox path for display (workdir/foo -> foo)."""
    root = os.path.abspath(root)
    if abs_path == root:
        return "."
    if abs_path.startswith(root + os.sep):
        return abs_path[len(root) + 1 :]
    return abs_path
