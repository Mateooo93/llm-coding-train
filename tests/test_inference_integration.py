"""Tests for the dual-format action parser (Ornith + native) and the OpenAI-compatible backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.agent import (
    Step,
    ToolCall,
    _parse_any_tool_call_block,
    _remap_file_editor,
    _openai_dict_to_local,
    format_action,
    parse_output,
    parse_model_response,
    strip_reasoning,
)
from src.inference.openai_client import (
    CompletionResult,
    OpenAIBackend,
    ToolCallRecord,
    ToolSpec,
    _parse_chat_response,
)


# ── Dual-format parser tests ────────────────────────────────


def test_literal_action_parses_basic_kwarg_pair():
    thought, action = parse_output("<thought>let's run ls</thought><action>bash(command='ls -la')</action>")
    assert thought == "let's run ls"
    assert action.tool == "bash"
    assert action.args == {"command": "ls -la"}


def test_literal_action_handles_multiple_kwargs():
    _, action = parse_output("<action>write(path='/tmp/x.py', content='hello world')</action>")
    assert action.tool == "write"
    assert action.args == {"path": "/tmp/x.py", "content": "hello world"}


def test_literal_action_handles_escape_in_string():
    _, action = parse_output(r"""<action>bash(command='echo it\'s ok')</action>""")
    assert action.args == {"command": "echo it's ok"}


def test_json_tool_call_parses_ornith_style():
    """The exact JSON structure Ornith emits inside  tool_call ... tool_call."""
    text = (
        "<thought>Need to inspect /etc/passwd</thought>\n"
        "<tool_call>\n"
        '{"name": "file_editor", "arguments": {"command": "view", "path": "/etc/passwd"}}\n'
        "</tool_call>"
    )
    thought, action = parse_output(text)
    assert thought == "Need to inspect /etc/passwd"
    # file_editor view → maps to local 'read'
    assert action.tool == "read"
    assert action.args == {"path": "/etc/passwd"}
    assert action.openai_tool_name == "file_editor"


def test_json_tool_call_parses_bash_directly():
    text = (
        "<tool_call>\n"
        '{"name": "bash", "arguments": {"command": "ls -la"}}\n'
        "</tool_call>"
    )
    _, action = parse_output(text)
    assert action.tool == "bash"
    assert action.args == {"command": "ls -la"}


def test_json_tool_call_handles_format_variants():
    """Ornith sometimes drops the closing tag or wraps in fences."""
    cases = [
        # No closing tag (model forgot)
        '<tool_call>{"name": "bash", "arguments": {"command": "pwd"}}',
        # Wrapped in ```json ... ``` fences
        'Then:\n```json\n{"name": "bash", "arguments": {"command": "pwd"}}\n```\n',
        # Double close-tag with whitespace
        '< tool_call >\n{"name": "bash", "arguments": {"command": "pwd"}}\n< / tool_call >',
    ]
    for c in cases:
        _, action = parse_output(c)
        assert action is not None, f"failed to parse: {c!r}"
        assert action.tool == "bash", f"parsed wrong tool: {c!r}"
        assert action.args == {"command": "pwd"}, f"wrong args: {c!r}"


def test_file_editor_create_remaps_to_write():
    text = (
        '<tool_call>{"name": "file_editor", "arguments": '
        '{"command": "create", "path": "/tmp/new.py", "file_text": "print(1)"}}\n'
        "</tool_call>"
    )
    _thought, action = parse_output(text)
    assert action.tool == "write"
    assert action.args == {"path": "/tmp/new.py", "content": "print(1)"}


def test_file_editor_str_replace_remaps_to_edit():
    text = (
        '<tool_call>{"name": "file_editor", "arguments": '
        '{"command": "str_replace", "path": "/tmp/x", "old_str": "old", "new_str": "new"}}\n'
        "</tool_call>"
    )
    _thought, action = parse_output(text)
    assert action.tool == "edit"
    assert action.args == {"path": "/tmp/x", "find": "old", "replace": "new"}


def test_finish_action_via_openai():
    text = '<tool_call>{"name": "finish", "arguments": {"answer": "Task complete."}}</tool_call>'
    _thought, action = parse_output(text)
    assert action.tool == "finish"
    assert action.args == {"answer": "Task complete."}


def test_unknown_tool_name_surfaces_in_observation():
    """We don't silently drop unknown tools; the executor reports them as errors."""
    text = '<tool_call>{"name": "nuke_datacenter", "arguments": {}}</tool_call>'
    _thought, action = parse_output(text)
    assert action.tool == "__unknown__"
    assert action.args == {"_model_name": "nuke_datacenter", "_model_args": {}}


def test_literal_format_takes_priority_over_json_within_same_text():
    """If the model emits both formats, prefer the literal one (matches our prompt)."""
    text = (
        "<thought>both</thought>\n"
        "<action>bash(command='literal')</action>\n"
        '<tool_call>{"name": "bash", "arguments": {"command": "from_json"}}</tool_call>'
    )
    _thought, action = parse_output(text)
    assert action.tool == "bash"
    assert action.args == {"command": "literal"}


def test_format_action_round_trips_through_parse():
    original = ToolCall(tool="write", args={"path": "/tmp/a.py", "content": "def f():\n    return 1\n"})
    text = format_action(original.tool, original.args)
    _thought, parsed_back = parse_output(text)
    assert parsed_back.tool == original.tool
    assert parsed_back.args == original.args


# ── Reasoning-strip tests ───────────────────────────────────


def test_strip_reasoning_removes_think_block():
    text = "This is reasoning that should not be visible to the user.\nThe actual final answer is here."
    out = strip_reasoning(text)
    assert "reasoning" not in out
    assert "actual final answer is here" in out


def test_strip_reasoning_handles_no_tags():
    assert strip_reasoning("plain text") == "plain text"


def test_strip_reasoning_handles_alternate_tag_name():
    """Ornith sometimes uses ``<thinking>`` instead of ``<reasoning>``."""
    text = "<thinking>hidden plan</thinking>visible reply"
    out = strip_reasoning(text)
    assert out == "visible reply"


# ── OpenAIBackend tests (no live HTTP) ──────────────────────


def test_openai_backend_uses_bearer_when_api_key_supplied():
    backend = OpenAIBackend(base_url="http://x/v1", model_id="m", api_key="sk-test")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.headers)
        m = MagicMock()
        m.read.return_value = b'{"choices":[{"message":{"content":"hi"}}]}'
        m.__enter__ = lambda s: s
        m.__exit__ = lambda s, *a: False
        return m

    with patch("src.inference.openai_client.urlrequest.urlopen", side_effect=fake_urlopen):
        backend.complete([{"role": "user", "content": "hello"}], max_tokens=10)
    assert captured["headers"].get("Authorization") == "Bearer sk-test"


def test_openai_backend_omits_authorization_when_no_api_key():
    backend = OpenAIBackend(base_url="http://x/v1", model_id="m", api_key=None)
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.headers)
        m = MagicMock()
        m.read.return_value = b'{"choices":[{"message":{"content":"hi"}}]}'
        m.__enter__ = lambda s: s
        m.__exit__ = lambda s, *a: False
        return m

    with patch("src.inference.openai_client.urlrequest.urlopen", side_effect=fake_urlopen):
        backend.complete([{"role": "user", "content": "hello"}], max_tokens=10)
    assert "Authorization" not in captured["headers"]


def test_react_agent_sends_tools_spec_to_http_backend():
    """Verify the agent uses OpenHands-style tools param when driving an HTTP backend."""
    from src.inference.agent import ReActAgent, ReActConfig

    fake = OpenAIBackend(base_url="http://x/v1", model_id="m")
    captured_body: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        captured_body.update(body)
        m = MagicMock()
        m.read.return_value = b'{"choices":[{"message":{"content":"OK","tool_calls":[{"id":"c","type":"function","function":{"name":"finish","arguments":"{\\"answer\\":\\"done\\"}"}}]}}]}'
        m.__enter__ = lambda s: s
        m.__exit__ = lambda s, *a: False
        return m

    agent = ReActAgent(model=fake, config=ReActConfig(max_steps=1, use_uncertainty=False))
    with patch("src.inference.openai_client.urlrequest.urlopen", side_effect=fake_urlopen):
        agent.run("test task")

    assert "tools" in captured_body, "tools=[] param not sent to LM Studio"
    tool_names = {t["function"]["name"] for t in captured_body["tools"]}
    assert {"bash", "file_editor", "finish"}.issubset(tool_names)


def test_react_agent_uses_openhands_system_prompt_for_http_backend():
    """When the backend is HTTP (Ornith / Devstral), the prompt should use OpenHands tool names."""
    from src.inference.agent import ReActAgent, ReActConfig, OPENHANDS_SYSTEM_PROMPT

    fake = OpenAIBackend(base_url="http://x/v1", model_id="m")
    agent = ReActAgent(model=fake, config=ReActConfig(max_steps=1))
    assert agent.system_prompt == OPENHANDS_SYSTEM_PROMPT
    assert "file_editor" in agent.system_prompt
    assert "<action>" not in agent.system_prompt


def test_react_agent_uses_native_prompt_for_attnreslm_backend():
    """AttnResLM still uses the literal <action> prompt format."""
    from src.inference.agent import ReActAgent, ReActConfig, REACT_SYSTEM_PROMPT

    # Build a fake AttnResLM-like object: no `complete` method
    class FakeAttnResLM:
        def forward(self, *a, **k):
            pass
        def parameters(self):
            return iter([])

    agent = ReActAgent(model=FakeAttnResLM(), config=ReActConfig(max_steps=1))
    assert agent.system_prompt == REACT_SYSTEM_PROMPT
    assert "<action>" in agent.system_prompt


def test_react_agent_uses_server_tool_calls_directly_not_reparse():
    """For HTTP backends, the agent should use server-returned tool_calls, not synth+reparse."""
    from src.inference.agent import ReActAgent, ReActConfig

    fake = OpenAIBackend(base_url="http://x/v1", model_id="m")

    server_response = b'{"choices":[{"message":{"content":"","reasoning_content":"thinking","tool_calls":[{"id":"abc","type":"function","function":{"name":"bash","arguments":"{\\"command\\":\\"pwd\\"}"}}]}}]}'

    def fake_urlopen(req, timeout=None):
        m = MagicMock()
        m.read.return_value = server_response
        m.__enter__ = lambda s: s
        m.__exit__ = lambda s, *a: False
        return m

    agent = ReActAgent(model=fake, config=ReActConfig(max_steps=1, use_uncertainty=False))
    with patch("src.inference.openai_client.urlrequest.urlopen", side_effect=fake_urlopen):
        result = agent.run("pwd please")

    assert result["success"] is True
    assert "test/pass" in result["answer"] or result["steps"][0].action.tool == "bash"
    assert result["steps"][-1].action.tool == "finish"


def test_parse_chat_response_extracts_tool_calls():
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "Let me list the directory first.",
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"ls"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }
    parse_chat_response = _parse_chat_response  # local alias for clarity
    result = _parse_chat_response(raw)
    assert isinstance(result, CompletionResult)
    assert result.reasoning == "Let me list the directory first."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].arguments == {"command": "ls"}
    assert result.tool_calls[0].id == "call_xyz"


def test_parse_chat_response_handles_no_choices():
    """Empty / filtered response should not crash the caller."""
    result = _parse_chat_response({"choices": []})
    assert result.content == ""
    assert result.tool_calls == []
    assert result.finish_reason == "empty"


def test_parse_chat_response_handles_malformed_arguments_gracefully():
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": "not-json"}},
                    ],
                }
            }
        ]
    }
    result = _parse_chat_response(raw)
    # Falls back to raw-string preservation
    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].arguments == {"_raw": "not-json"}


def test_parse_model_response_uses_native_tool_calls_when_present():
    """When the server returns tool_calls but content is empty, parse_model_response should still yield a ToolCall."""
    result = CompletionResult(
        content="",
        reasoning="Let me call bash.",
        tool_calls=[
            ToolCallRecord(id="abc", name="bash", arguments={"command": "ls"}),
        ],
    )
    thought, action, reasoning = parse_model_response(result)
    assert reasoning == "Let me call bash."
    assert action is not None
    assert action.tool == "bash"
    assert action.args == {"command": "ls"}
    assert action.tool_call_id == "abc"


def test_parse_chat_response_serialization_for_tools():
    """Round-trip: ToolSpec → OpenAI dict inside the request body."""
    spec = ToolSpec(
        name="bash",
        description="Run a shell command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    from src.inference.openai_client import _tool_spec_to_openai
    d = _tool_spec_to_openai(spec)
    assert d["type"] == "function"
    assert d["function"]["name"] == "bash"
    assert d["function"]["parameters"]["required"] == ["command"]


# ── File-editor de-multiplex helper ────────────────────────


def test_file_editor_view_remaps():
    tc = _remap_file_editor("file_editor", {"command": "view", "path": "/etc/hosts"})
    assert tc.tool == "read"
    assert tc.args == {"path": "/etc/hosts"}


def test_str_replace_editor_insert_uses_anchor_field():
    tc = _remap_file_editor("str_replace_editor", {"command": "insert", "path": "/x", "new_str": "X"})
    assert tc.tool == "edit"
    assert tc.args["replace"] == "X"


def test_openai_dict_to_local_handles_stringified_args_legacy():
    """Some servers (older LM Studio) send arguments as a JSON string. We still parse."""
    payload = {"name": "bash", "arguments": '{"command":"pwd"}'}
    tc = _openai_dict_to_local(payload)
    assert tc is not None
    assert tc.args == {"command": "pwd"}
