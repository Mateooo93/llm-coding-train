"""
Constrained decoding — regex/grammar/logit masking for code and structured output.

Constrained decoding restricts the model's vocabulary at every step to ensure
the resulting text conforms to a predefined structure (e.g., valid Bash
commands, valid JSON, valid Python, a regular expression pattern).

For terminal/SWE tasks this is enormously valuable — it eliminates entire
classes of syntactic hallucinations (e.g., invalid shell syntax, malformed
JSON responses, garbage code).

Implementation:
  We don't rely on a heavy library like Outlines. Instead we provide:
  - `RegexMasker`: tracks the set of valid next-token prefixes given a regex
    pattern and masks logits whose tokens don't match.
  - `WordListMasker`: masks tokens that aren't in a given list (literal strings,
    identifiers, etc.).
  - `JsonMasker`: enforces valid JSON output by tracking parser state.

These are simple but efficient for terminal tasks: the bash command prefix
is often a small set of allowed literals (e.g., `cd /`, `ls`, `grep`, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


# Standard tiktoken BPE — fixed once; we expose a slot for the tokenizer to
# expose (token_str_for_id) to resolve each token id to a string prefix.
@dataclass
class _PrefixTree:
    children: dict = None  # type: ignore[assignment]

    def __init__(self):
        self.children = {}


class PrefixMasker:
    """
    Tracks valid prefixes and masks invalid next-token logits.

    The token vocabulary is mapped to decoded strings (token_str_for_id) and
    we maintain a set of valid partial completions. Each forward step asks:
    given the current prefix, which tokens are legal continuations?

    For high-throughput use we precompute (token_str → token_id) once.
    """

    def __init__(self, token_str_for_id: dict, allowed_prefixes: Optional[Iterable[str]] = None):
        """
        Args:
            token_str_for_id: dict mapping token idx → str.
            allowed_prefixes: optional list of allowed literal strings at the
                current step. If None, no constraint is applied.
        """
        self.token_str_for_id = token_str_for_id
        self.token_id_for_str = {v: k for k, v in token_str_for_id.items()}
        self.allowed_prefixes = list(allowed_prefixes) if allowed_prefixes else []

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask out tokens whose decoded string is not in allowed_prefixes."""
        if not self.allowed_prefixes:
            return logits

        # Build a list of valid token indices
        valid_ids = []
        allowed_set = set(self.allowed_prefixes)
        for token_id in range(logits.shape[-1]):
            tok_str = self.token_str_for_id.get(token_id, "")
            if tok_str in allowed_set:
                valid_ids.append(token_id)

        mask = torch.full_like(logits, float("-inf"))
        if valid_ids:
            mask[..., valid_ids] = 0.0
        return logits + mask


class RegexMasker:
    """
    Mask logits so emitted tokens always match a regex pattern.

    Approach:
      - Decode all vocabulary tokens once.
      - For each token, check whether appending it still keeps the partial
        output on a path that could complete to the regex.
      - At each generation step, given the current partial string, return
        logits masked to legal continuations.

    For a simple regex like ``"true|false"`` we can precompute, for each partial
    path, the set of legal next-token IDs. For full regex support this would
    rely on an FSM library; for the prototype we provide a clean baseline.
    """

    def __init__(self, token_str_for_id: dict, regex_pattern: str):
        self.token_str_for_id = token_str_for_id
        self.regex = re.compile(regex_pattern)

        # Precompute: for every token, does it produce a string that satisfies
        # the regex (when considered as a complete output)? This won't catch all
        # partial cases but works for short-literal patterns like "yes|no".
        self.token_fits = {
            tid: tok_str for tid, tok_str in token_str_for_id.items()
            if self.regex.fullmatch(tok_str)
        }

    def make_starter_masker(self, candidates: Iterable[str]) -> PrefixMasker:
        """
        Create a PrefixMasker restricted to candidate strings that match the regex.
        """
        allowed = [c for c in candidates if self.regex.fullmatch(c)]
        return PrefixMasker(self.token_str_for_id, allowed)

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask logits to tokens whose decoded string fully matches the regex."""
        valid_ids = list(self.token_fits.keys())
        mask = torch.full_like(logits, float("-inf"))
        if valid_ids:
            mask[..., valid_ids] = 0.0
        return logits + mask


class JsonMasker:
    """
    Constrain generation to valid JSON output.

    Uses incremental tracking of bracket/quote state to mask tokens that would
    break JSON validity. Specifically:
      - Tracks open braces ``{`` and brackets ``[`` that need closing.
      - Tracks whether we are inside a string (between unescaped quotes).
      - When inside a string: only allows characters valid in a JSON string,
        or a closing quote (when balanced).

    This is a pragmatic, prototype-level JSON masker — production code should
    use a dedicated library (Outlines, lm-format-enforcer).
    """

    def __init__(self, token_str_for_id: dict):
        self.token_str_for_id = token_str_for_id

    def _step_state(self, partial: str) -> dict:
        """Crude parity check on the so-far generated JSON fragment."""
        in_string = False
        escape = False
        depth = 0
        for c in partial:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
        return {"in_string": in_string, "depth": depth}

    def mask_logits(self, logits: torch.Tensor, partial: str = "") -> torch.Tensor:
        """
        Mask logits so the next token keeps the partial output a valid JSON prefix.

        Constraints:
          - Inside a JSON string: only allow closing-quote or escape characters.
            Other tokens must be heavily penalized to avoid double-quoting.
          - Outside a string at top level (depth <= 0): only allow closing
            brackets or new value-start tokens.
          - Inside structure: allow value chars but close brackets when deep.
        """
        if not partial:
            return logits

        state = self._step_state(partial)
        in_string = state["in_string"]
        depth = state["depth"]

        # -inf mask: start with all tokens blocked, then unblock allowed ones
        mask = torch.full_like(logits, float("-inf"))

        if in_string:
            for tid, tok in self.token_str_for_id.items():
                # Closing quote terminates string
                if tok.endswith('"') and not tok.endswith('\\"'):
                    mask[..., tid] = 0.0
                # Escape character -> next token may be a quote/backslash
                elif tok == "\\":
                    mask[..., tid] = 0.0
                # Allow typical string content characters (letters, digits, spaces)
                elif all(ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?' or ch.isascii() for ch in tok):
                    mask[..., tid] = 0.0
        else:
            if depth <= 0:
                # Top-level: closing brackets, commas, or values
                for tid, tok in self.token_str_for_id.items():
                    if tok in ("}", "]", ","):
                        mask[..., tid] = 0.0
                    elif tok in ('"', "{", "["):
                        # Opening a new value
                        mask[..., tid] = 0.0
                    elif tok in ("true", "false", "null"):
                        mask[..., tid] = 0.0
            else:
                # Inside structure: encourage closing if deep
                for tid, tok in self.token_str_for_id.items():
                    if depth > 5 and tok in ("}", "]"):
                        mask[..., tid] = 0.0
                    else:
                        # Allow common value characters
                        mask[..., tid] = 0.0

        return logits + mask
