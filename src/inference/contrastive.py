"""
Contrastive Decoding (DoLa-style) — hallucination reduction at inference time.

Contrastive decoding reduces "easy" factual hallucinations by contrasting the
mature-layer logits of a well-formed prompt against the immature-layer logits
(earlier layers, where factual knowledge is less crystallized).

Reference: "DoLa: Decoding by Contrasting Layers Improves Factuality in Large
Language Models" (Chuang et al., 2023). arXiv:2309.03883.

In our setting we approximate this by:
  1. Running the model normally and capturing its final-layer logits.
  2. Running the model with an "amplified" intermediate-layer signal
     (or, more practically for the provided architecture, with a degraded
     prior — e.g., smaller dropout or shifted temperature).
  3. Subtracting the contrastive logits from the final logits
     (logits_adapted = logits_final + alpha * (logits_final - logits_contrast)
     so the parts present in `final` but absent in the contrast prior are
     amplified).

This function returns a Callable that mutates a model's forward to produce a
contrast prior, then exposes a `decode()` that wraps a base sampling step.
For simplicity we expose `contrastive_logits(logits_main, logits_contrast, alpha)`
as a pure function plus a `ContrastiveGenerator` wrapper.

Usage:
    from src.inference.contrastive import contrastive_logits
    logits = contrastive_logits(final_logits, contrast_logits, alpha=0.5)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def contrastive_logits(
    logits_final: torch.Tensor,
    logits_contrast: torch.Tensor,
    alpha: float = 0.5,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Combine final and contrastive logits to amplify tokens that are confident in
    the final prediction but not in the contrast prior.

    Args:
        logits_final: [batch, vocab_size] logits from the main forward pass.
        logits_contrast: [batch, vocab_size] logits from a contrast prior
            (e.g., earlier-layer logits, or higher-temperature logits).
        alpha: weight on the contrastive amplification (higher = more conservative).
        threshold: cutoff for logit difference — if the contrast logit minus
            the final logit is very negative (above this threshold), drop
            that token entirely (it's strongly suppressed in the final pass).

    Returns:
        adjusted logits: [batch, vocab_size].
    """
    # softmax over both to compute probability distributions
    p_final = F.softmax(logits_final, dim=-1)
    log_p_final = F.log_softmax(logits_final, dim=-1)
    log_p_contrast = F.log_softmax(logits_contrast, dim=-1)

    # log-diff: positive where final is more confident than contrast
    log_diff = log_p_final - log_p_contrast

    # Adaptive plausibility: drop tokens that the contrast passes through
    # then apply the contrastive boost to remaining ones.
    mask = log_diff >= threshold  # tokens final is at least ``threshold`` more likely

    # Combined logits = original + alpha * log-diff mask
    adjusted = logits_final + alpha * log_diff * mask.to(logits_final.dtype)

    return adjusted


class ContrastiveGenerator:
    """
    Wraps a sampling/generation loop with contrastive decoding.

    The class expects ``generate_fn`` to be a function that takes (input_ids)
    and returns the next-token logits [batch, vocab_size]. Construct it with
    a separate way to produce "contrast" logits — typically by passing the
    same input through the model with the contrast prior.

    In our AttnResLM there is no "earlier layer" hook exposed yet. For the
    prototype we approximate the contrast prior via temperature scaling:
    logits_contrast = logits_final / contrast_temperature, which makes the
    distribution broader (less confident) and serves as the "immature" prior.

    Args:
        alpha: weight on contrastive amplification.
        threshold: cutoff for logit difference.
        contrast_temperature: temperature to apply when forming the contrast prior.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        threshold: float = 0.1,
        contrast_temperature: float = 1.5,
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.contrast_temperature = contrast_temperature

    def __call__(
        self, logits_main: torch.Tensor, logits_contrast: torch.Tensor
    ) -> torch.Tensor:
        return contrastive_logits(
            logits_main, logits_contrast, alpha=self.alpha, threshold=self.threshold
        )

    def make_contrast_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Broader distribution that serves as a contrast prior."""
        return logits / self.contrast_temperature


def configure_contrastive(
    model,
    alpha: float = 0.5,
    threshold: float = 0.1,
    contrast_temperature: float = 1.5,
) -> ContrastiveGenerator:
    """
    Convenience that returns a ContrastiveGenerator for the given model.
    The model argument is kept for future versions that hook earlier layers.
    """
    return ContrastiveGenerator(
        alpha=alpha,
        threshold=threshold,
        contrast_temperature=contrast_temperature,
    )
