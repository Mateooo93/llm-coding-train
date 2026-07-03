"""
SwiGLU (Swish-Gated Linear Unit) MLP.

SwiGLU has become the standard FFN activation in modern LLMs (Llama, Mistral, Gemma),
offering superior expressivity per parameter compared to standard GeLU or ReLU FFNs.

The gating mechanism:
    FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down

where SiLU is the Sigmoid Linear Unit: SiLU(x) = x * sigmoid(x).

The intermediate dimension is typically ~8/3 * hidden_size (rounded to a multiple
of a power of 2 for hardware efficiency) to maintain parameter parity with a
standard 4x FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP with three projections: gate, up, and down.

    Unlike a standard FFN with 2 projections (up, down), SwiGLU uses 3:
        - gate_proj:  d → d_ff  (followed by SiLU activation)
        - up_proj:    d → d_ff  (element-wise multiplied with gate output)
        - down_proj:  d_ff → d  (projects back to hidden dimension)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        gate = F.silu(self.gate_proj(x))  # SiLU(x @ W_gate)
        up = self.up_proj(x)               # x @ W_up
        return self.down_proj(gate * up)   # (gate * up) @ W_down
