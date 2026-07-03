"""
Text generation for the AttnRes LM.

Supports:
  - Greedy decoding
  - Temperature sampling
  - Top-k sampling
  - Top-p (nucleus) sampling
  - KV-cache-free generation (simple, for prototype)

For production, this would use a KV-cache for efficient autoregressive generation,
but the simple loop is sufficient for validating the model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..model import AttnResLM


@torch.no_grad()
def generate_text(
    model: AttnResLM,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate text from a prompt using the AttnRes LM.

    Args:
        model: The AttnResLM model.
        prompt: Input text prompt.
        tokenizer: tiktoken Encoding.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (1.0 = no change, <1 = more conservative).
        top_k: If set, only sample from the top-k highest probability tokens.
        top_p: Nucleus sampling threshold (keep tokens with cumulative prob >= top_p).
        device: Device to run generation on.

    Returns:
        Generated text string (including the prompt).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )  # [1, seq_len]

    generated = input_ids

    for _ in range(max_new_tokens):
        # Truncate to max position embeddings if needed
        if generated.shape[1] > model.config.max_position_embeddings:
            generated = generated[:, -model.config.max_position_embeddings:]

        # Forward pass
        output = model(generated)
        logits = output.logits[:, -1, :]  # [1, vocab_size] — last token logits

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1:]
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample next token
        if temperature == 0:
            # Greedy decoding
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Stop if end-of-text token is generated
        if next_token.item() == tokenizer.eot_token:
            break

    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text


@torch.no_grad()
def generate_completion(
    model: AttnResLM,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate a completion (returns only the new text, not the prompt).
    """
    full_text = generate_text(
        model, prompt, tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )
    return full_text[len(prompt):]
