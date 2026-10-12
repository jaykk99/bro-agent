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
