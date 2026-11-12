"""Tests for the app/tool_base.py fallback shim.

tool_base is the only app module with no dedicated tests, yet the whole
dependency-free suite runs on it: when `uniharness` is not installed (local
test/CI machines, this VM), every tool class inherits the fallback
BaseAgentTool and returns the fallback ToolResult. If the shim drifts from
the interface the tools and the agent factory use (name/description/
args_schema attrs, async execute(params) -> ToolResult with .output,
__call__ turning bad args into error output instead of raising), every
tool test goes red at once — or worse, passes here and breaks on Render.

These tests pin the fallback contract. They only run when the fallback is
active (uniharness absent); on a machine with uniharness installed the
real classes are imported instead and this module skips.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pydantic import BaseModel

from app import tool_base
from app.tool_base import BaseAgentTool, ToolResult

pytestmark = pytest.mark.skipif(
    tool_base.USING_UNIHARNESS,
    reason="fallback shim tests only apply when uniharness is absent",
)


class EchoParams(BaseModel):
    text: str
    shout: bool = False


class EchoTool(BaseAgentTool[EchoParams]):
    name = "echo"
    description = "Echoes text back, optionally upper-cased."
    args_schema = EchoParams

    async def execute(self, params):
        out = params.text.upper() if params.shout else params.text
        return ToolResult(output=out)


class NoSchemaTool(BaseAgentTool):
    name = "no_schema"
    description = "Takes raw kwargs, no validation."

    async def execute(self, params):
        return ToolResult(output=f"got {sorted(params.items())}")


def test_fallback_is_active_without_uniharness():
    assert tool_base.USING_UNIHARNESS is False


def test_tool_result_defaults_to_empty_output():
    assert ToolResult().output == ""
    assert ToolResult(output="hi").output == "hi"


def test_base_defaults():
    tool = BaseAgentTool()
    assert tool.name == "unnamed_tool"
    assert tool.description == ""
    assert tool.args_schema is None


def test_base_execute_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        asyncio.run(BaseAgentTool().execute(None))


def test_init_kwargs_override_class_attrs():
    tool = EchoTool(name="renamed", description="custom")
    assert tool.name == "renamed"
    assert tool.description == "custom"
    assert tool.args_schema is EchoParams


def test_run_validates_and_executes():
    out = asyncio.run(EchoTool().run(text="hello")).output
    assert out == "hello"
    out = asyncio.run(EchoTool().run(text="hello", shout=True)).output
    assert out == "HELLO"


def test_run_rejects_invalid_kwargs():
    with pytest.raises(Exception):
        asyncio.run(EchoTool().run())


def test_call_validates_and_executes():
    out = asyncio.run(EchoTool()(text="hey")).output
    assert out == "hey"


def test_call_returns_error_output_instead_of_raising():
    # The agent sees a bad tool call as tool output, not a crash: the
    # turn survives and the model can retry with fixed args.
    result = asyncio.run(EchoTool()())
    assert isinstance(result, ToolResult)
    assert "echo" in result.output
    assert "failed" in result.output.lower()


def test_call_without_schema_passes_kwargs_dict_through():
    result = asyncio.run(NoSchemaTool()(a=1, b=2))
    assert result.output == "got [('a', 1), ('b', 2)]"


def test_run_without_schema_passes_kwargs_dict_through():
    result = asyncio.run(NoSchemaTool().run(a=1))
    assert result.output == "got [('a', 1)]"


def test_real_tool_class_drives_through_fallback_call():
    # A shipped tool (FileReadTool) under the fallback: valid args run,
    # missing required args come back as error output, never an exception.
    from app.tools_files import FileReadTool

    tool = FileReadTool()
    assert tool.name == "file_read"
    assert tool.args_schema is not None

    missing = asyncio.run(tool(path="/nonexistent/nope.txt"))
    assert isinstance(missing, ToolResult)
    assert missing.output  # tool reports the missing file as output

    bad_args = asyncio.run(tool())
    assert "file_read" in bad_args.output
    assert "failed" in bad_args.output.lower()
