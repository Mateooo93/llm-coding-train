"""
ReAct Agent scaffold — Reason + Act loop for terminal / SWE / coding agents.

ReAct (Reason + Act) interleaves natural-language reasoning with tool calls.
At each step the agent produces a Thought (chain-of-thought reasoning) and
then an Action (a structured tool call). The environment executes the
action and returns an Observation, which feeds into the next Thought.

Two backend types are supported:
  (A) :class:`AttnResLM` — our own model, called via :func:`generate_text`
  (B) :class:`OpenAIBackend` — any OpenAI-compatible HTTP server
        (LM Studio, vLLM, Modal). The HTTP server returns tool calls as JSON
        via the OpenAI ``tool_calls`` array, and may emit `` reasoning... ``
        blocks before the final answer (reasoning models like Ornith).

Two action formats are supported, parsed in priority order:
  (1) In-line literal: ``<action>bash(command='ls')</action>``  (our default prompt)
  (2) OpenAI JSON:    `` tool_call
        {"name": "bash", "arguments": {"command": "ls"}}
       tool_call ``  (Ornith / OpenHands / Devstral native)

A ``file_editor`` / ``str_replace_editor`` tool call from the model is
de-multiplexed into our local tools:
  - command="view"     -> ``read``
  - command="create"   -> ``write``
  - command="str_replace" -> ``edit``
  - command="insert"   -> ``edit`` (treated as find-and-replace by anchor)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

from ..model import AttnResLM
from .openai_client import CompletionResult, ModelBackend, OpenAIBackend, ToolCallRecord
from .uncertainty import evaluate_uncertainty


# ── Data structures ─────────────────────────────────────────


@dataclass
class ToolCall:
    """A parsed action from the model's output."""
    tool: str  # local tool name: 'bash', 'read', 'write', 'edit', 'finish'
    args: dict = field(default_factory=dict)
    # Optional provenance — set when the call was parsed from OpenAI JSON
    openai_tool_name: Optional[str] = None   # e.g. "file_editor" if remapped
    tool_call_id: Optional[str] = None


@dataclass
class Step:
    """One step in the ReAct loop."""
    thought: str
    action: ToolCall
    observation: str
    reasoning: Optional[str] = None  # raw  ...  block, preserved for debugging


# ── Tool implementations ─────────────────────────────────────

DEFAULT_DENYLIST = [
    r"\brm\s+-rf\s+/\b",
    r"\bdd\s+if=.*of=/dev/",
    r"\bmkfs\b",
    r":\(\)\{\s*:\|\:\s*&\s*\};:",
    r"\bcurl\s+.*\s*\|\s*sh\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\b\s+0",
]


class BashTool:
    REQUIRED_KEYS = ("command",)

    def __init__(
        self,
        cwd: Optional[str] = None,
        timeout: int = 30,
        denylist: Optional[list] = None,
    ):
        self.cwd = cwd
        self.timeout = timeout
        self.history: list = []
        self.denylist = denylist if denylist is not None else DEFAULT_DENYLIST

    def run(self, command: str) -> str:
        if not isinstance(command, str):
            return f"[error: 'command' must be a string, got {type(command).__name__}]"
        for pattern in self.denylist:
            if re.search(pattern, command):
                return f"[refused: command matches safety pattern '{pattern}']"
        self.history.append(command)
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=self.timeout, cwd=self.cwd,
            )
            output = result.stdout
            if result.stderr:
                output += ("\nSTDERR:\n" if output else "") + result.stderr
            if result.returncode != 0:
                output += f"\n[exit code {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"[timeout after {self.timeout}s]"
        except Exception as e:
            return f"[error: {e}]"


class FileReadTool:
    def run(self, path: str) -> str:
        try:
            with open(path, "r") as f:
                content = f.read()
            return content if len(content) < 8000 else content[:8000] + "\n[truncated]"
        except Exception as e:
            return f"[error: {e}]"


class FileWriteTool:
    def run(self, path: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"[wrote {len(content)} bytes to {path}]"
        except Exception as e:
            return f"[error: {e}]"


class FileEditTool:
    def run(self, path: str, find: str, replace: str) -> str:
        try:
            with open(path, "r") as f:
                content = f.read()
            if find not in content:
                return f"[error: '{find[:50]}...' not found in {path}]"
            new_content = content.replace(find, replace)
            with open(path, "w") as f:
                f.write(new_content)
            return f"[edited {path}]"
        except Exception as e:
            return f"[error: {e}]"


TOOLS = {
    "bash": BashTool,
    "read": FileReadTool,
    "write": FileWriteTool,
    "edit": FileEditTool,
}


# ── Reasoning / tool-call format helpers ────────────────────


THOUGHT_PATTERN = re.compile(r"<thought>(?P<thought>.*?)</thought>", re.DOTALL)
REASONING_PATTERN = re.compile(r"<(?:think|reasoning)>(?P<reasoning>.*?)</(?:think|reasoning)>", re.DOTALL)

LITERAL_ACTION_PATTERN = re.compile(
    r"<action>\s*(?P<tool>\w+)\((?P<args>.*?)\)\s*</action>",
    re.DOTALL,
)

# Robust JSON tool-call pattern. Handles `` tool_call { ... } tool_call``,
# optional whitespace, optional double-quoted/ single-quoted JSON, and the
# case where the model forgets the closing tag.
TOOL_CALL_PATTERN = re.compile(
    r"<\s*tool[_\s]?call\s*>(?P<body>.*?)(?:<\s*/\s*tool[_\s]?call\s*>|$)",
    re.DOTALL | re.IGNORECASE,
)

# OpenHands tool-name -> our local tool name. file_editor / str_replace_editor
# are de-multiplexed by their ``command`` argument, so the mapping is a function,
# not a dict.
OPENHANDS_TOOL_TO_LOCAL: dict[str, str] = {
    "bash": "bash",
    "execute_bash": "bash",
    "terminal": "bash",
    "finish": "finish",
    "complete": "finish",
}
# When the model emits ``file_editor`` / ``str_replace_editor`` we dispatch by command
MULTI_COMMAND_TOOLS = {"file_editor", "str_replace_editor", "editor"}


# ── Action parser / formatter ────────────────────────────────


def format_action(tool: str, args: dict) -> str:
    """Format a ToolCall dict back into the model's output language (literal form)."""
    arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    return f"<action>{tool}({arg_str})</action>"


def parse_output(text: str) -> tuple[Optional[str], Optional[ToolCall]]:
    """Extract thought + action from the model's raw output.

    Tries, in order:
      1. ``<thought>...</thought><action>tool(args)</action>`` (our prompt format)
      2. `` tool_call {...} tool_call`` with JSON
    """
    thought_match = THOUGHT_PATTERN.search(text)
    thought = thought_match.group("thought").strip() if thought_match else None

    # (1) Literal action
    literal = LITERAL_ACTION_PATTERN.search(text)
    if literal is not None:
        tool = literal.group("tool")
        raw_args = literal.group("args").strip()
        return thought, ToolCall(tool=tool, args=_parse_kw_args(raw_args))

    # (2) OpenAI-style JSON tool call
    tc = _parse_any_tool_call_block(text)
    if tc is not None:
        return thought, tc

    return thought, None


def parse_model_response(
    result: CompletionResult,
) -> tuple[Optional[str], Optional[ToolCall], Optional[str]]:
    """Parse a CompletionResult from OpenAIBackend.

    Returns (thought, action, reasoning_raw):
      - thought: best-effort extracted Thought from content or chain-of-thought
      - action: ToolCall parsed from either tool_calls or in-content tool_call block
      - reasoning_raw: the unstripped reasoning trace (preserved for debugging)
    """
    content = result.content or ""
    reasoning = result.reasoning

    # If the server returned native tool_calls, prefer them
    if result.tool_calls:
        # Use the first non-empty call (one-step policies only care about one)
        for r in result.tool_calls:
            tc = _openai_record_to_local(r)
            if tc is not None:
                # reasoning → use reasoning_content if present, otherwise content
                thought_text = _extract_thought_from_text(content) or (reasoning or "")
                return thought_text, tc, reasoning

    # Fallback: parse content as text the same way as a raw backend string
    thought, action = parse_output(content)
    return thought, action, reasoning


# ── Internal helpers ─────────────────────────────────────────


class _LiteralArgParser:
    """Safe (no ``eval``) parser for ``key='value', key2=42``-style arguments."""

    def parse(self, raw: str) -> dict:
        out: dict = {}
        if not raw.strip():
            return out
        tokens = self._tokenize(raw)
        if not tokens:
            return out
        i = 0
        while i < len(tokens):
            k = tokens[i]
            i += 1
            if i < len(tokens) and tokens[i] == "=":
                i += 1
                v, i = self._parse_value(tokens, i)
                out[k] = v
            else:
                out[k] = True
        return out

    def _tokenize(self, raw: str) -> list:
        tokens = []
        i, n = 0, len(raw)
        while i < n:
            c = raw[i]
            if c.isspace() or c == ",":
                i += 1
                continue
            if c in "'\"":
                quote = c
                j = i + 1
                buf = []
                while j < n:
                    if raw[j] == "\\" and j + 1 < n:
                        buf.append(raw[j + 1])
                        j += 2
                        continue
                    if raw[j] == quote:
                        break
                    buf.append(raw[j])
                    j += 1
                tokens.append("".join(buf))
                i = j + 1
                continue
            j = i
            while j < n and not raw[j].isspace() and raw[j] not in ",=()[]{}":
                j += 1
            tokens.append(raw[i:j])
            i = j
        return tokens

    def _parse_value(self, tokens: list, i: int) -> tuple:
        if i >= len(tokens):
            return None, i
        t = tokens[i]
        if t == "True":
            return True, i + 1
        if t == "False":
            return False, i + 1
        if t == "None":
            return None, i + 1
        try:
            return int(t), i + 1
        except ValueError:
            pass
        try:
            return float(t), i + 1
        except ValueError:
            pass
        return t, i + 1


def _parse_kw_args(raw: str) -> dict:
    return _LiteralArgParser().parse(raw)


def _parse_any_tool_call_block(text: str) -> Optional[ToolCall]:
    """Find a JSON tool-call block (Ornith / OpenHands / Devstral format)."""
    match = TOOL_CALL_PATTERN.search(text)
    if match is None:
        return None
    body = match.group("body").strip()
    # Strip surrounding ```json fences if the model wrapped the JSON
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        # Last-ditch: scan all top-level {...} blocks via raw extraction
        for cand in re.findall(r"\{[\s\S]*?\}", body):
            try:
                payload = json.loads(cand)
                break
            except json.JSONDecodeError:
                continue
        else:
            return None
    return _openai_dict_to_local(payload)


def _openai_record_to_local(rec: ToolCallRecord) -> Optional[ToolCall]:
    """Maps an :class:`ToolCallRecord` (server-returned tool_calls[]) to a local ToolCall."""
    payload = {"name": rec.name, "arguments": rec.arguments}
    tc = _openai_dict_to_local(payload)
    if tc is not None:
        tc.tool_call_id = rec.id or None
    return tc


def _openai_dict_to_local(payload: dict) -> Optional[ToolCall]:
    """Maps a parsed OpenAI-style tool-call dict ``{name, arguments}`` to a local ToolCall.

    Handles OpenHands sub-tool disambiguation (file_editor's ``command`` argument).
    """
    name = payload.get("name") or payload.get("function") or ""
    args = payload.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_value": args}

    if name in MULTI_COMMAND_TOOLS:
        return _remap_file_editor(name, args)
    local_name = OPENHANDS_TOOL_TO_LOCAL.get(name)
    if local_name is None:
        # Unknown tool — surface it so we can debug the mapping rather than silently drop
        return ToolCall(tool="__unknown__", args={"_model_name": name, "_model_args": args}, openai_tool_name=name)
    return ToolCall(tool=local_name, args=args, openai_tool_name=name)


def _remap_file_editor(model_tool_name: str, args: dict) -> ToolCall:
    """Disambiguate OpenHands file_editor command into our read/write/edit tools."""
    command = (args.get("command") or "").lower().strip()
    if command == "view":
        return ToolCall(
            tool="read",
            args={"path": args.get("path") or args.get("file_text") or ""},
            openai_tool_name=model_tool_name,
        )
    if command == "create":
        return ToolCall(
            tool="write",
            args={"path": args.get("path", ""), "content": args.get("file_text") or args.get("content", "")},
            openai_tool_name=model_tool_name,
        )
    if command in ("str_replace", "insert"):
        return ToolCall(
            tool="edit",
            args={
                "path": args.get("path", ""),
                "find": args.get("old_str") or args.get("find", ""),
                "replace": args.get("new_str") or args.get("replace", ""),
            },
            openai_tool_name=model_tool_name,
        )
    # Unknown subcommand — pass through and let the tool fail loudly
    return ToolCall(
        tool="__unknown__",
        args={"_model_name": model_tool_name, "_command": command, "_model_args": args},
        openai_tool_name=model_tool_name,
    )


def _extract_thought_from_text(text: str) -> Optional[str]:
    """Pull out a ``<thought>...</thought>`` block if present in `text`."""
    m = THOUGHT_PATTERN.search(text)
    return m.group("thought").strip() if m else None


def strip_reasoning(content: str) -> str:
    """Remove `` reasoning... `` blocks from a model's output, returning the post-think content."""
    return REASONING_PATTERN.sub("", content).strip()


# ── ReAct prompt builder ─────────────────────────────────────


REACT_SYSTEM_PROMPT = """You are an expert AI agent that solves software-engineering tasks by reasoning step-by-step and acting on a real Linux environment.

At each step, produce text in EXACTLY the following format:

<thought>
Your natural-language reasoning. Explain what you know, what you need to find out, and which tool will help.
</thought>
<action>
tool(arg1='value1', arg2='value2')
</action>

When the task is complete, use the special tool ``finish(answer='...')`` to terminate.

Available tools:
- bash(command='shell command') — Run a shell command. Returns stdout, stderr, and exit code.
- read(path='/abs/path') — Read a file. Returns content (truncated to 8000 chars).
- write(path='/abs/path', content='...') — Write a string to a file.
- edit(path='/abs/path', find='exact block', replace='new block') — Replace a block in a file.
- finish(answer='your final answer to the user') — Terminate when done.
"""

# OpenHands-style system prompt for backends whose base model was trained on the
# OpenHands harness (Ornith-1.0, Devstral-Small-2505). These models expect tool
# names ``bash`` / ``file_editor`` / ``finish`` with structured arguments.
OPENHANDS_SYSTEM_PROMPT = """You are an expert AI agent that solves software-engineering tasks by reasoning step-by-step and acting on a real Linux environment.

Use the tool definitions provided by the system. At each step, emit either:
- A natural-language plan describing your reasoning, OR
- A tool call using one of the available functions.

Available tools (exact names):
- bash(command: string) — Run a shell command. Returns stdout, stderr, exit code.
- file_editor(command, path, ...) — Inspect or modify a file. Sub-commands:
    * view(path)
    * create(path, file_text)
    * str_replace(path, old_str, new_str)
    * insert(path, new_str)
- finish(answer: string) — Terminate with the final answer.

Make exactly ONE tool call per turn. When the task is solved, call finish.
"""


@dataclass
class ReActConfig:
    max_steps: int = 12
    max_per_step_tokens: int = 400
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    use_uncertainty: bool = True


# ── Agent ─────────────────────────────────────────────────────


def _is_openai_backend(model: Any) -> bool:
    return isinstance(model, OpenAIBackend)


class ReActAgent:
    """ReAct loop over an AttnResLM or any OpenAI-compatible backend."""

    def __init__(
        self,
        model: Union[AttnResLM, OpenAIBackend],
        tokenizer=None,
        config: Optional[ReActConfig] = None,
        cwd: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer  # only required when using AttnResLM
        self.config = config or ReActConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

        self.tools = {
            "bash": BashTool(cwd=cwd, timeout=30),
            "read": FileReadTool(),
            "write": FileWriteTool(),
            "edit": FileEditTool(),
        }
        self.history: list = []
        self._uses_http_backend = _is_openai_backend(model)
        # Pick the system prompt that matches the model's training.
        # - HTTP backend (Ornith / Devstral / Qwen): trained on OpenHands, expects
        #   structured tool-call JSON, not the literal ``<action>tool(args)</action>``.
        # - AttnResLM: trained on our literal prompt format.
        # If the caller explicitly passed a custom system_prompt, respect it.
        if system_prompt is None:
            self.system_prompt = OPENHANDS_SYSTEM_PROMPT if self._uses_http_backend else REACT_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt

    # ── Public entry ─────────────────────────────────────────

    def run(self, task: str) -> dict:
        transcript = f"=== TASK ===\n{task}\n\n=== STEPS ===\n"
        success = False
        final_answer: Optional[str] = None
        uncertainty_flags = []

        for step_num in range(self.config.max_steps):
            step = self._take_step(transcript)
            self.history.append(step)
            transcript = self._append_step_to_transcript(transcript, step_num, step)
            uncertainty_flags.append(int(getattr(step, "uncertain", False)))

            if step.action.tool == "finish":
                final_answer = step.action.args.get("answer", "")
                success = True
                break

        return {
            "answer": final_answer or "[no finish action]",
            "steps": self.history,
            "success": success,
            "uncertainty_flags": uncertainty_flags,
            "transcript": transcript,
        }

    # ── One step ─────────────────────────────────────────────

    def _take_step(self, transcript: str) -> Step:
        prompt = self._build_prompt(transcript)

        if self._uses_http_backend:
            # HTTP path: use the server's tool_calls directly when the model is
            # OpenHands-trained. We do not re-parse the assistant content as text.
            result = self._generate_via_http(transcript)
            thought, action, reasoning = parse_model_response(result)
        else:
            # Local path: append ``<thought>`` stub and parse the literal action.
            text_full = self._generate_via_local(prompt, prompt + "\n<thought>")
            thought, action = parse_output(text_full)
            reasoning = None

        if action is None:
            action = ToolCall(tool="finish", args={"answer": "(could not parse action)"})

        obs = ""
        if action.tool != "finish":
            obs = self._execute(action)
        step = Step(thought=thought or "", action=action, observation=obs)
        step.uncertain = False
        step.reasoning = reasoning
        return step

    def _build_tool_specs(self) -> list[ToolSpec]:
        """Tool definitions sent to the OpenAI backend. Mirrors OpenHands schema."""
        return [
            ToolSpec(
                name="bash",
                description="Run a shell command and return stdout, stderr, and exit code.",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            ),
            ToolSpec(
                name="file_editor",
                description="Inspect or modify a local file. Use the command argument to choose the operation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]},
                        "path": {"type": "string"},
                        "file_text": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                    },
                    "required": ["command", "path"],
                },
            ),
            ToolSpec(
                name="finish",
                description="Terminate the loop with the final answer.",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
        ] 

    # ── Backend dispatchers ─────────────────────────────────

    def _generate_via_local(self, prompt: str, prompt_for_generation: str) -> str:
        from .generate import generate_text  # local import to keep module order
        text = generate_text(
            self.model,
            prompt_for_generation,
            self.tokenizer,
            max_new_tokens=self.config.max_per_step_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        text_full = text[len(prompt):].strip() if text.startswith(prompt) else text
        text_full = "<thought>" + text_full
        m = re.search(r"</action>", text_full)
        if m:
            text_full = text_full[: m.end()]
        return text_full

    def _generate_via_http(self, transcript: str) -> CompletionResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": transcript.rstrip()},
        ]
        # Send the tool schema so LM Studio / vLLM activates its tool-call parser
        # and returns a structured `tool_calls` array. This is what Ornith was
        # trained to emit — without it the server replies with raw `` tool_call ``
        # text we have to re-parse by hand.
        result: CompletionResult = self.model.complete(  # type: ignore[union-attr]
            messages,
            tools=self._build_tool_specs(),
            tool_choice="auto",
            max_tokens=self.config.max_per_step_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stop=["</action>", "Observation:"],
        )
        return result

    # ── Tool execution ──────────────────────────────────────

    def _execute(self, action: ToolCall) -> str:
        if action.tool == "__unknown__":
            return f"[error: model emitted unknown tool {action.openai_tool_name!r}: {action.args}]"
        tool = self.tools.get(action.tool)
        if tool is None:
            return f"[error: unknown tool '{action.tool}']"
        if hasattr(tool, "REQUIRED_KEYS"):
            missing = [k for k in tool.REQUIRED_KEYS if k not in action.args]
            if missing:
                return f"[error: missing required args: {missing}]"
        try:
            return tool.run(**action.args)
        except Exception as e:
            return f"[error: {type(e).__name__}: {e}]"

    # ── Prompt assembly ─────────────────────────────────────

    def _build_prompt(self, transcript: str) -> str:
        return f"{self.system_prompt}\n\n{transcript}"

    @staticmethod
    def _append_step_to_transcript(transcript: str, step_num: int, step: Step) -> str:
        block = (
            f"\n--- Step {step_num + 1} ---\n"
            f"<thought>{step.thought}</thought>\n"
            f"<action>{step.action.tool}("
            + ", ".join(f"{k}={repr(v)}" for k, v in step.action.args.items())
            + ")</action>\n"
        )
        if step.action.tool != "finish":
            block += f"<observation>{step.observation}</observation>\n"
        return transcript + block
