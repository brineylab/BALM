# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
from torch import nn

__all__ = [
    # "RelativePositionalEmbedding",
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


# class RelativePositionalEmbedding(nn.Module):
#     """
#     Relative positional embeddings, as initially described in the
#     `T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_ paper.

#     Parameters
#     ----------
#     num_heads: int
#         The number of heads.

#     max_position: int
#         The maximum position.

#     .. _T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer:
#         https://arxiv.org/abs/1910.10683

#     """

#     def __init__(self, num_heads: int, max_position: int = 512):
#         super().__init__()
#         self.num_heads = num_heads
#         self.max_position = max_position
#         self.relative_positions = nn.Embedding(2 * max_position - 1, num_heads)
#         self.register_buffer(
#             "positions", torch.arange(-max_position + 1, max_position), persistent=False
#         )

#     def forward(self, seq_len: int) -> torch.Tensor:
#         """
#         Apply relative positional embeddings to the input tensor.

#         Parameters
#         ----------
#         seq_len: int
#             The sequence length.

#         Returns
#         -------
#         torch.Tensor
#             The input tensor with relative positional embeddings applied. The shape is (batch, seq_len, num_heads).

#         """
#         range_vec = torch.arange(seq_len, device=self.relative_positions.weight.device)
#         relative_positions_matrix = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
#         relative_positions_matrix = relative_positions_matrix + (self.max_position - 1)
#         relative_positions_matrix = relative_positions_matrix.clamp(
#             0, 2 * self.max_position - 2
#         )
#         rel_embeddings = self.relative_positions(relative_positions_matrix)
#         return rel_embeddings
