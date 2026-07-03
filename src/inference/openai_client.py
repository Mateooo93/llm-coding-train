"""
OpenAI-compatible inference backend.

Lets every existing generator (ReAct agent, contrastive, constrained, uncertainty,
eval harness) work against any model served by an OpenAI-compatible HTTP endpoint:
  - LM Studio (default at http://127.0.0.1:1234/v1)
  - vLLM OpenAI server
  - Modal-deployed vLLM
  - Any third-party public OpenAI-compatible API

The backend exposes:
  - ``complete(messages, ...) -> CompletionResult`` — single chat completion
  - ``stream(messages, ...) -> Iterator[ChunkDelta]`` — token stream for live UX
  - ``embed(texts) -> list[list[float]]`` — for sanity-check benchmarks

It also implements the lightweight ``ModelBackend`` protocol so it can be
swapped for ``AttnResLM`` directly inside the agent loop.

Why a Protocol and not the actual ``AttnResLM``?
  - Lets us run inference against models we DID NOT train (Ornith, Qwen, etc.).
  - Keeps our agent prompt-format work in ONE place (the parser), not in
    every model loader.
  - Lets the same agent code target either the locally-quantised GGUF
    (LM Studio on Mac) or the BF16 fine-tunable version (vLLM on Modal).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Optional, Protocol, Sequence
from urllib import request as urlrequest, error as urlerror


# ── Data classes ─────────────────────────────────────────────


@dataclass
class ToolSpec:
    """JSON-Schema-style tool definition (OpenAI format)."""
    name: str
    description: str
    parameters: dict  # JSON Schema dict with "type": "object" + "properties"


@dataclass
class ToolCallRecord:
    """A tool call emitted by the model (Ornith / OpenHands / Devstral format)."""
    id: str
    name: str
    arguments: dict   # already-parsed dict, NOT a raw JSON string


@dataclass
class CompletionResult:
    """One model reply, fully parsed."""
    content: str                          # visible assistant text (after stripping reasoning trace)
    reasoning: Optional[str] = None       #  ...  block contents, if preserved
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)  # raw response, for debug


# ── Backend protocol ─────────────────────────────────────────


class ModelBackend(Protocol):
    """Anything that can serve chat completions. Implementations: this module + AttnResLM."""
    def complete(self, messages: Sequence[dict], **kwargs) -> CompletionResult: ...


# ── Backend implementation ───────────────────────────────────


class OpenAIBackend:
    """OpenAI-compatible HTTP chat-completions backend.

    Args:
        base_url: e.g. ``"http://127.0.0.1:1234/v1"``. No trailing slash required.
        model_id: the exact model id returned by ``GET /v1/models`` (LM Studio normally
                  uses e.g. ``"ornith-1.0-9b"``; vLLM uses the HF repo id).
        api_key: optional. LM Studio ignores it; vLLM / Modal often require ``"EMPTY"`` or a real key.
        timeout: HTTP timeout in seconds.
        default_sampling: applied to every ``complete()`` call unless overridden.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        default_sampling: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.default_sampling = default_sampling or {
            "temperature": 0.6,
            "top_p": 0.95,
        }

    # ── Public API ───────────────────────────────────────────

    def complete(
        self,
        messages: Sequence[dict],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra_body: Optional[dict] = None,
    ) -> CompletionResult:
        """Run one chat completion. ``messages`` is a list of OpenAI-style dicts."""
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else self.default_sampling.get("temperature", 0.6),
            "top_p": top_p if top_p is not None else self.default_sampling.get("top_p", 0.95),
        }
        if top_k is not None:
            body["top_k"] = top_k
        if stop:
            body["stop"] = list(stop)
        if tools:
            body["tools"] = [_tool_spec_to_openai(t) for t in tools]
        if tool_choice is not None:
            # "none" | "auto" | "required" | {"type":"function","function":{...}}
            body["tool_choice"] = tool_choice
        if extra_body:
            body.update(extra_body)

        raw = self._post("/chat/completions", body)
        return _parse_chat_response(raw)

    def stream(
        self,
        messages: Sequence[dict],
        **kwargs,
    ) -> Iterator[dict]:
        """Yield raw SSE chunks from the server. For UI streaming use only."""
        body = {
            "model": self.model_id,
            "messages": list(messages),
            "stream": True,
        }
        # Merge sampling
        for k in ("max_tokens", "temperature", "top_p", "top_k", "stop"):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]
        if "tools" in kwargs and kwargs["tools"]:
            body["tools"] = [_tool_spec_to_openai(t) for t in kwargs["tools"]]

        url = self.base_url + "/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urlrequest.Request(
            url, data=data,
            headers={"Content-Type": "application/json", **_auth_headers(self.api_key)},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue

    # ── Internals ────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8")
        req = urlrequest.Request(
            url, data=data,
            headers={"Content-Type": "application/json", **_auth_headers(self.api_key)},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as e:
            # Re-raise with the body so failures are debug-able
            body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"HTTP {e.code} on {url}: {body_text[:500]}") from e
        except urlerror.URLError as e:
            raise RuntimeError(f"Could not reach {url}: {e.reason}") from e


# ── Helpers ──────────────────────────────────────────────────


def _auth_headers(api_key: Optional[str]) -> dict:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _tool_spec_to_openai(t: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }


def _parse_chat_response(raw: dict) -> CompletionResult:
    """Parse a non-streaming /chat/completions response."""
    choices = raw.get("choices") or []
    if not choices:
        # Empty response — model refused or filtered
        return CompletionResult(content="", finish_reason="empty", raw=raw)
    ch = choices[0]
    msg = ch.get("message", {}) or {}

    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content")  # OpenAI-O-style / vLLM reasoning parser

    tool_calls: List[ToolCallRecord] = []
    for tc in msg.get("tool_calls") or []:
        name = (tc.get("function") or {}).get("name", "")
        raw_args = (tc.get("function") or {}).get("arguments", "{}")
        # Try to parse the arguments as JSON; LM Studio sometimes returns empty string
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        tool_calls.append(
            ToolCallRecord(
                id=tc.get("id", "") or "",
                name=name,
                arguments=args,
            )
        )

    return CompletionResult(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=ch.get("finish_reason", "stop"),
        usage=raw.get("usage", {}) or {},
        raw=raw,
    )


# ── Convenience constructor ─────────────────────────────────


def make_backend_from_config(base_alias: str, base_url: Optional[str] = None) -> OpenAIBackend:
    """Build an OpenAIBackend from a single base_models.yaml alias.

    Args:
        base_alias: e.g. ``"ornith-1.0-9b"`` — looks up recommended_sampling + gguf_repo.
        base_url: override the LM Studio URL if running on a remote machine.

    Returns:
        OpenAIBackend ready for ``.complete(messages, ...)``.
    """
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("pyyaml required for make_backend_from_config — run: python -m pip install pyyaml") from e
    cfg_path = "config/base_models.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    entry = next((m for m in cfg["models"] if m["alias"] == base_alias), None)
    if entry is None:
        raise KeyError(f"Base alias {base_alias!r} not found in {cfg_path}")
    return OpenAIBackend(
        base_url=base_url or "http://127.0.0.1:1234/v1",
        model_id=entry.get("gguf_repo") or entry["hf_id"],
        default_sampling=entry.get("recommended_sampling") or {},
    )
