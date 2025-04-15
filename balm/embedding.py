# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
from torch import nn

__all__ = [
    "RotaryPositionalEmbedding",
]


def apply_rotary_emb(q_or_k, sin, cos):
    """
    Apply rotary positional embeddings to the input tensor.

    Parameters
    ----------
    q_or_k: torch.Tensor
        The input tensor. Expected shape is (batch_size, seq_len, num_heads, head_dim).
    sin: torch.Tensor
        The sine of the positional embeddings.
    cos: torch.Tensor
        The cosine of the positional embeddings.

    Returns
    -------
    torch.Tensor
        The input tensor with rotary positional embeddings applied.
        The shape is (batch_size, seq_len, num_heads, head_dim).
    """

    q1, q2 = q_or_k.chunk(2, dim=-1)  # ==> (..., head_dim/2)
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embeddings, as initially described in the
    `RoFormer: Enhanced Transformer with Rotary Position Embeddings`_ paper.

    Parameters
    ----------
    dim: int
        The embedding dimension.
    base: int
        The base of the exponential function.

    References
    ----------
    .. _RoFormer: Enhanced Transformer with Rotary Position Embeddings:
        https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # frequencies for the "half-dim"
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 1).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x.shape = (batch_size, seq_len, num_heads, dim)
        positions.shape = (batch_size, seq_len) or None
        """
        bsz, seq_len, num_heads, head_dim = x.shape
        # if no positions given, assume 0..seq_len-1 for each example
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
            positions = positions.unsqueeze(0).expand(bsz, seq_len)  # (b, s)

        # shape: (b, s, half_dim)
        angles = positions.unsqueeze(-1).to(x.dtype) * self.inv_freq.unsqueeze(
            0
        ).unsqueeze(0)
        # or use torch.einsum('bs, d -> bsd', positions, self.inv_freq)

        # broadcast across heads => shape (b, s, num_heads, half_dim)
        angles = angles.unsqueeze(2)  # (b, s, 1, half_dim)
        angles = angles.expand(-1, -1, num_heads, -1)

        sin, cos = angles.sin(), angles.cos()
        return apply_rotary_emb(x, sin, cos)
