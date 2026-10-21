"""Tool base class with a graceful fallback when UniHarness isn't installed.

In production (Render) `uniharness` is installed and the real
`BaseAgentTool` is used. For local testing / CI without the heavy
UniHarness dependency, a minimal compatible fallback is defined so the
tool modules (and their tests) still import.

The fallback mirrors the interface the agent factory uses:
  * class attributes: name, description, args_schema (a pydantic model)
  * async execute(params) -> ToolResult-like object with an `output` attr
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

try:  # production path
    from uniharness.tools import BaseAgentTool as _RealBase  # type: ignore
    from uniharness.types import ToolResult as _RealResult  # type: ignore

    BaseAgentTool = _RealBase  # noqa: F401
    ToolResult = _RealResult  # noqa: F401
    USING_UNIHARNESS = True
except Exception:  # local test path — no uniharness installed

    from pydantic import BaseModel

    USING_UNIHARNESS = False

    class ToolResult(BaseModel):  # type: ignore[no-redef]
        """Minimal stand-in: carries the tool's text output."""

        output: str = ""

    _P = TypeVar("_P", bound=BaseModel)

    class BaseAgentTool(Generic[_P]):  # type: ignore[no-redef]
        """Minimal stand-in for uniharness.tools.BaseAgentTool.

        Subclasses set name/description/args_schema and implement
        `async def execute(self, params) -> ToolResult`.
        """

        name: str = "unnamed_tool"
        description: str = ""
        args_schema: Any = None

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        async def execute(self, params: Any) -> ToolResult:  # noqa: ANN401
            raise NotImplementedError

        async def run(self, **kwargs: Any) -> ToolResult:
            """Convenience: validate kwargs against args_schema, then execute."""
            schema = getattr(self, "args_schema", None)
            params = schema(**kwargs) if schema is not None else kwargs
            return await self.execute(params)
