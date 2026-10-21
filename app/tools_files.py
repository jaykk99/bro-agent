"""File tools for the agent — read/write/edit/list/search, all jailed to BRO_WORKDIR.

The agent previously had no file tools of its own (it relied on whatever
UniHarness's computer exposed). These give it precise, auditable file
operations with guardrails: path escapes are impossible (app/sandbox.py),
reads are capped, writes are capped, edits require an exact-match anchor,
and binary files are refused rather than mangled.
"""

from __future__ import annotations

import os
import re
from typing import Any

from pydantic import BaseModel, Field

from .sandbox import SandboxViolation, display_path, resolve, workdir
from .tool_base import BaseAgentTool, ToolResult

READ_MAX_BYTES = 200 * 1024  # 200 KB per read
WRITE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per write
LIST_MAX_ENTRIES = 500
SEARCH_MAX_MATCHES = 100


def _binary_sniff(data: bytes) -> bool:
    return b"\x00" in data[:8192]


class FileReadParams(BaseModel):
    path: str = Field(description="Path to read, relative to the workdir (e.g. 'notes/todo.txt')")
    max_bytes: int = Field(default=READ_MAX_BYTES, description="Max bytes to return")


class FileReadTool(BaseAgentTool[FileReadParams]):
    name: str = "file_read"
    description: str = "Read a text file from the workdir. Returns its content (capped)."
    args_schema = FileReadParams

    async def execute(self, params: FileReadParams) -> ToolResult:
        root = workdir()
        try:
            p = resolve(root, params.path)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        if not os.path.isfile(p):
            return ToolResult(output=f"Not a file: {display_path(root, p)}")
        cap = max(1, min(params.max_bytes, READ_MAX_BYTES))
        try:
            with open(p, "rb") as f:
                data = f.read(cap + 1)
        except OSError as e:
            return ToolResult(output=f"Read failed: {e}")
        if _binary_sniff(data):
            return ToolResult(output=f"Binary file, not shown: {display_path(root, p)}")
        truncated = len(data) > cap
        text = data[:cap].decode("utf-8", errors="replace")
        note = "\n[... truncated ...]" if truncated else ""
        return ToolResult(output=f"{display_path(root, p)}:\n{text}{note}")


class FileWriteParams(BaseModel):
    path: str = Field(description="Path to write, relative to the workdir")
    content: str = Field(description="Full new content of the file")


class FileWriteTool(BaseAgentTool[FileWriteParams]):
    name: str = "file_write"
    description: str = "Write (create or overwrite) a text file in the workdir."
    args_schema = FileWriteParams

    async def execute(self, params: FileWriteParams) -> ToolResult:
        root = workdir()
        try:
            p = resolve(root, params.path)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        data = params.content.encode("utf-8")
        if len(data) > WRITE_MAX_BYTES:
            return ToolResult(output=f"Blocked: content exceeds {WRITE_MAX_BYTES} byte cap")
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(params.content)
        except OSError as e:
            return ToolResult(output=f"Write failed: {e}")
        return ToolResult(output=f"Wrote {len(data)} bytes to {display_path(root, p)}")


class FileEditParams(BaseModel):
    path: str = Field(description="Path to edit, relative to the workdir")
    old_text: str = Field(description="Exact text to find (must occur exactly once)")
    new_text: str = Field(description="Replacement text")


class FileEditTool(BaseAgentTool[FileEditParams]):
    name: str = "file_edit"
    description: str = (
        "Surgical edit: replace one exact occurrence of old_text with new_text. "
        "Fails if the anchor text occurs zero times or more than once — "
        "re-read the file and use a longer anchor."
    )
    args_schema = FileEditParams

    async def execute(self, params: FileEditParams) -> ToolResult:
        root = workdir()
        try:
            p = resolve(root, params.path)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        if not os.path.isfile(p):
            return ToolResult(output=f"Not a file: {display_path(root, p)}")
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            return ToolResult(output=f"Read failed: {e}")
        count = text.count(params.old_text)
        if count == 0:
            return ToolResult(output="Anchor text not found — re-read the file and copy it exactly.")
        if count > 1:
            return ToolResult(output=f"Anchor text occurs {count} times — use a longer, unique anchor.")
        text = text.replace(params.old_text, params.new_text, 1)
        if len(text.encode("utf-8")) > WRITE_MAX_BYTES:
            return ToolResult(output="Blocked: result would exceed the write size cap")
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            return ToolResult(output=f"Write failed: {e}")
        return ToolResult(output=f"Edited {display_path(root, p)}")


class FileListParams(BaseModel):
    path: str = Field(default=".", description="Directory to list, relative to the workdir")
    show_hidden: bool = Field(default=False, description="Include dotfiles")


class FileListTool(BaseAgentTool[FileListParams]):
    name: str = "file_list"
    description: str = "List a directory in the workdir (files, dirs, sizes)."
    args_schema = FileListParams

    async def execute(self, params: FileListParams) -> ToolResult:
        root = workdir()
        try:
            p = resolve(root, params.path)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        if not os.path.isdir(p):
            return ToolResult(output=f"Not a directory: {display_path(root, p)}")
        try:
            entries = sorted(os.listdir(p))
        except OSError as e:
            return ToolResult(output=f"List failed: {e}")
        lines = []
        for name in entries:
            if name.startswith(".") and not params.show_hidden:
                continue
            full = os.path.join(p, name)
            try:
                if os.path.isdir(full) and not os.path.islink(full):
                    lines.append(f"{name}/")
                else:
                    size = os.path.getsize(full)
                    lines.append(f"{name} ({size} bytes)")
            except OSError:
                lines.append(f"{name} (?)")
            if len(lines) >= LIST_MAX_ENTRIES:
                lines.append(f"... truncated at {LIST_MAX_ENTRIES} entries")
                break
        return ToolResult(output="\n".join(lines) if lines else "(empty)")


class FileSearchParams(BaseModel):
    pattern: str = Field(description="Regex to search for")
    path: str = Field(default=".", description="File or directory to search, relative to the workdir")
    case_sensitive: bool = Field(default=False, description="Case-sensitive match")


class FileSearchTool(BaseAgentTool[FileSearchParams]):
    name: str = "file_search"
    description: str = "Grep-like regex search across text files in the workdir."
    args_schema = FileSearchParams

    async def execute(self, params: FileSearchParams) -> ToolResult:
        root = workdir()
        try:
            p = resolve(root, params.path)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        try:
            rx = re.compile(params.pattern, 0 if params.case_sensitive else re.I)
        except re.error as e:
            return ToolResult(output=f"Bad regex: {e}")
        targets: list[str] = []
        if os.path.isfile(p):
            targets = [p]
        elif os.path.isdir(p):
            for dirpath, _dirnames, filenames in os.walk(p):
                for fn in filenames:
                    if fn.startswith("."):
                        continue
                    targets.append(os.path.join(dirpath, fn))
                if len(targets) > 2000:
                    break
        else:
            return ToolResult(output=f"Not found: {display_path(root, p)}")
        hits: list[str] = []
        for t in targets:
            try:
                if os.path.getsize(t) > READ_MAX_BYTES:
                    continue
                with open(t, "rb") as f:
                    data = f.read()
                if _binary_sniff(data):
                    continue
                for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{display_path(root, t)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= SEARCH_MAX_MATCHES:
                            break
            except OSError:
                continue
            if len(hits) >= SEARCH_MAX_MATCHES:
                break
        if not hits:
            return ToolResult(output="No matches.")
        note = f"\n... capped at {SEARCH_MAX_MATCHES} matches" if len(hits) >= SEARCH_MAX_MATCHES else ""
        return ToolResult(output="\n".join(hits) + note)


class FileMkdirParams(BaseModel):
    path: str = Field(description="Directory to create, relative to the workdir")


class FileMkdirTool(BaseAgentTool[FileMkdirParams]):
    name: str = "file_mkdir"
    description: str = "Create a directory (and parents) in the workdir."
    args_schema = FileMkdirParams

    async def execute(self, params: FileMkdirParams) -> ToolResult:
        root = workdir()
        try:
            p = resolve(root, params.path)
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")
        try:
            os.makedirs(p, exist_ok=True)
        except OSError as e:
            return ToolResult(output=f"mkdir failed: {e}")
        return ToolResult(output=f"Created {display_path(root, p)}/")


def file_tools() -> list[Any]:
    return [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        FileListTool(),
        FileSearchTool(),
        FileMkdirTool(),
    ]
