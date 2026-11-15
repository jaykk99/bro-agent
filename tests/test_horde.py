"""Tests for the Horde uncensored fallback (no network needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.horde_llm import HordeChatModel, _parse_tool_calls, _messages_to_prompt
from langchain_core.messages import HumanMessage, SystemMessage


def test_parse_fenced_tool_call():
    text = 'Checking now.\n```tool_call\n{"name": "web_search", "arguments": {"query": "x"}}\n```'
    clean, calls = _parse_tool_calls(text)
    assert calls == [{"name": "web_search", "args": {"query": "x"}}]
    assert "tool_call" not in clean
    assert "Checking now." in clean


def test_parse_multiple_tool_calls():
    text = ('```tool_call\n{"name": "a", "arguments": {}}\n```\n'
            '```tool_call\n{"name": "b", "arguments": {"k": 1}}\n```')
    _, calls = _parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[1] == {"name": "b", "args": {"k": 1}}


def test_parse_plain_reply():
    clean, calls = _parse_tool_calls("Just a normal reply.")
    assert calls == []
    assert clean == "Just a normal reply."


def test_parse_bad_json_ignored():
    clean, calls = _parse_tool_calls('```tool_call\nnot json at all\n```')
    assert calls == []


def test_parse_bare_json_nested_arguments():
    # Most natural bare form: same shape as the fenced format, no fence.
    # The old [^{}]* regex could not span the nested "arguments" object,
    # so this parsed to zero calls and the raw JSON was shown as reply text.
    text = 'Let me search.\n{"name": "web_search", "arguments": {"query": "winnipeg weather"}}'
    clean, calls = _parse_tool_calls(text)
    assert calls == [{"name": "web_search", "args": {"query": "winnipeg weather"}}]
    assert "web_search" not in clean
    assert "Let me search." in clean


def test_parse_bare_json_flat_form():
    clean, calls = _parse_tool_calls('{"name": "web_search", "query": "winnipeg"}')
    assert calls == [{"name": "web_search", "args": {"query": "winnipeg"}}]
    assert clean == ""


def test_parse_bare_json_skips_non_tool_objects():
    text = 'Config was {"foo": 1}. Now calling {"name": "file_read", "arguments": {"path": "a.txt"}}'
    clean, calls = _parse_tool_calls(text)
    assert calls == [{"name": "file_read", "args": {"path": "a.txt"}}]
    assert '{"foo": 1}' in clean


def test_bind_tools_injects_prompt():
    class FakeTool:
        name = "web_search"
        description = "Search the web."
        args = {"query": "string"}

    m = HordeChatModel()
    b = m.bind_tools([FakeTool()])
    assert isinstance(b, HordeChatModel)
    assert len(b._bound_tools) == 1
    # original untouched
    assert len(m._bound_tools) == 0
    prompt = b._build_prompt([SystemMessage(content="sys"), HumanMessage(content="hi")])
    assert "web_search" in prompt
    assert "Search the web." in prompt
    assert "```tool_call" in prompt
    assert "User: hi" in prompt
    assert prompt.rstrip().endswith("Assistant:")


def test_messages_to_prompt_tool_history():
    from langchain_core.messages import AIMessage, ToolMessage
    ai = AIMessage(content="looking",
                   tool_calls=[{"name": "web_search", "args": {"query": "x"},
                                "id": "call_1", "type": "tool_call"}])
    tm = ToolMessage(content="result!", name="web_search", tool_call_id="call_1")
    p = _messages_to_prompt([HumanMessage(content="q"), ai, tm], "")
    assert "Tool result (web_search): result!" in p
    assert '"name": "web_search"' in p


def test_llm_type():
    assert HordeChatModel()._llm_type == "horde-uncensored"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("ALL HORDE TESTS PASS")
