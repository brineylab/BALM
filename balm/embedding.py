# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import math
from typing import Optional

import torch
from torch import nn

__all__ = [
    "RelativePositionalEmbedding",
    "RotaryPositionalEmbedding",
]


def apply_rotary_emb(q_or_k, sin, cos):
    # q_or_k: (batch_size, seq_len, num_heads, head_dim)
    # split last dim into 2 halves
    q1, q2 = q_or_k.chunk(2, dim=-1)
    # each has shape (..., head_dim/2)
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)


class RotaryPositionalEmbedding(nn.Module):
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

        # shape = (b, s, half_dim)
        angles = positions.unsqueeze(-1).to(x.dtype) * self.inv_freq.unsqueeze(
            0
        ).unsqueeze(0)
        # or use torch.einsum('bs, d -> bsd', positions, self.inv_freq)

        # We want to broadcast across heads => shape (b, s, num_heads, half_dim)
        angles = angles.unsqueeze(2)  # (b, s, 1, half_dim)
        angles = angles.expand(-1, -1, num_heads, -1)

        sin, cos = angles.sin(), angles.cos()
        return apply_rotary_emb(x, sin, cos)


# class RotaryPositionalEmbedding(nn.Module):
#     """
#     Rotary positional embeddings, as initially described in the
#     `RoFormer: Enhanced Transformer with Rotary Position Embeddings`_ paper.

#     Parameters
#     ----------
#     dim: int
#         The embedding dimension.

#     .. _RoFormer: Enhanced Transformer with Rotary Position Embeddings:
#         https://arxiv.org/abs/2104.09864

#     """

#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
#         """
#         Apply rotary positional embeddings to the input tensor.

#         Parameters
#         ----------
#         x: torch.Tensor
#             The input tensor. Expected shape is (batch, seq_len, hidden_size).

#         seq_len: Optional[int]
#             The sequence length. If not provided, the sequence length is inferred from the input tensor.

#         Returns
#         -------
#         torch.Tensor
#             The input tensor with rotary positional embeddings applied. The shape is (batch, seq_len, hidden_size).

#         """
#         b, n, d = x.size()  # [batch, seq_len, hidden_size]
#         if seq_len is None:
#             seq_len = n
#         half_d = d // 2
#         position = torch.arange(seq_len, device=x.device).unsqueeze(1)
#         freqs = torch.exp(
#             -math.log(10000) * torch.arange(0, half_d, device=x.device).float() / half_d
#         )
#         angles = position * freqs.unsqueeze(0)
#         sin, cos = angles.sin(), angles.cos()

#         x1, x2 = x[..., :half_d], x[..., half_d:]
#         x_rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
#         return x_rotated  # [batch, seq_len, hidden_size]


class RelativePositionalEmbedding(nn.Module):
    """
    Relative positional embeddings, as initially described in the
    `T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_ paper.

    Parameters
    ----------
    num_heads: int
        The number of heads.

    max_position: int
        The maximum position.

    .. _T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer:
        https://arxiv.org/abs/1910.10683

    """

    def __init__(self, num_heads: int, max_position: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.max_position = max_position
        self.relative_positions = nn.Embedding(2 * max_position - 1, num_heads)
        self.register_buffer(
            "positions", torch.arange(-max_position + 1, max_position), persistent=False
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Apply relative positional embeddings to the input tensor.

        Parameters
        ----------
        seq_len: int
            The sequence length.

        Returns
        -------
        torch.Tensor
            The input tensor with relative positional embeddings applied. The shape is (batch, seq_len, num_heads).

        """
        range_vec = torch.arange(seq_len, device=self.relative_positions.weight.device)
        relative_positions_matrix = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
        relative_positions_matrix = relative_positions_matrix + (self.max_position - 1)
        relative_positions_matrix = relative_positions_matrix.clamp(
            0, 2 * self.max_position - 2
        )
        rel_embeddings = self.relative_positions(relative_positions_matrix)
        return rel_embeddings


# class RelativePositionalEmbedding(nn.Module):
#     """
#     Relative positional embeddings, as initially described in the
#     `Relative Position Embeddings for Transformers`_ paper.

#     Parameters
#     ----------
#     embed_dim: int
#         The embedding dimension.

#     max_length: int
#         The maximum length of the input tensor.

#     .. _Relative Position Embeddings for Transformers:
#         https://arxiv.org/abs/1803.02155

#     """

#     def __init__(self, embed_dim: int, max_length: int):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_length, embed_dim)
#         position = torch.arange(0, max_length).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, embed_dim, 2)
#             * -(torch.log(torch.tensor(10000.0)) / embed_dim)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply relative positional embeddings to the input tensor.

#         Parameters
#         ----------
#         x: torch.Tensor
#             The input tensor. Expected shape is (batch_size, seq_len, dim).

#         Returns
#         -------
#         torch.Tensor
#             The input tensor with relative positional embeddings applied. The shape is (batch_size, seq_len, dim).
#         """
#         x = x + self.pe[:, : x.size(1)]
#         return x


# class RotaryPositionalEmbedding(nn.Module):
#     """
#     Rotary positional embeddings, as initially described in the
#     `RoFormer: Enhanced Transformer with Rotary Position Embeddings`_ paper.

#     Parameters
#     ----------
#     embed_dim: int
#         The embedding dimension.

#     .. _RoFormer: Enhanced Transformer with Rotary Position Embeddings:
#         https://arxiv.org/abs/2104.09864

#     """

#     def __init__(self, embed_dim: int):
#         super().__init__()
#         self.embed_dim = embed_dim

#     def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
#         """
#         Apply rotary positional embeddings to the input tensor.

#         Parameters
#         ----------
#         x: torch.Tensor
#             The input tensor. Expected shape is (batch, seq_len, heads, head_dim).

#         seq_len: Optional[int]
#             The sequence length. If not provided, the sequence length is inferred from the input tensor.

#         Returns
#         -------
#         torch.Tensor
#             The input tensor with rotary positional embeddings applied. The shape is (batch, seq_len, heads, head_dim).

#         """
#         # x: (batch, seq_len, heads, head_dim)
#         b, n, h, d = x.size()
#         if seq_len is None:
#             seq_len = n
#         half_d = d // 2
#         position = torch.arange(seq_len, device=x.device).unsqueeze(1)
#         freqs = torch.exp(
#             -math.log(10000) * torch.arange(0, half_d, device=x.device).float() / half_d
#         )
#         angles = position * freqs.unsqueeze(0)
#         sin, cos = angles.sin(), angles.cos()

#         x1, x2 = x[..., :half_d], x[..., half_d:]
#         x_rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
#         return x_rotated


# class RotaryPositionalEmbedding(nn.Module):
#     """
#     Rotary positional embeddings, as initially described in the
#     `RoFormer: Enhanced Transformer with Rotary Position Embeddings`_ paper.

#     Parameters
#     ----------
#     embed_dim: int
#         The embedding dimension.

#     max_length: int
#         The maximum length of the input tensor.

#     .. _RoFormer: Enhanced Transformer with Rotary Position Embeddings:
#         https://arxiv.org/abs/2104.09864

#     """

#     def __init__(self, embed_dim: int, max_length: int):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.max_length = max_length
#         self.inv_freq = 1.0 / (
#             10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim)
#         )

#     def get_positional_embeddings(self, x: torch.Tensor):
#         """
#         Generates the sinusoidal positional embeddings.

#         Parameters
#         ----------
#         x: torch.Tensor
#             The input tensor. Expected shape is (batch_size, seq_len, dim).

#         Returns
#         -------
#         torch.Tensor
#             The positional embeddings. The shape is (seq_len, dim).
#         """
#         seq_len = x.shape[1]
#         positions = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
#         sinusoid_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
#         return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

#     def apply_rotary_embeddings(self, x: torch.Tensor):
#         """
#         Applies rotary embeddings to the input tensor x.

#         Parameters
#         ----------
#         x: torch.Tensor
#             The input tensor. Expected shape is (batch_size, seq_len, dim).

#         Returns
#         -------
#         torch.Tensor
#             The input tensor with rotary embeddings applied. The shape is (batch_size, seq_len, dim).

#         """
#         pos_emb = self.get_positional_embeddings(x).to(x.device)
#         s, c = pos_emb[:, : self.embed_dim // 2], pos_emb[:, self.embed_dim // 2 :]
#         x1, x2 = x[..., : self.embed_dim // 2], x[..., self.embed_dim // 2 :]
#         x_rot = torch.cat(((x1 * c) + (x2 * s), (-x1 * s) + (x2 * c)), dim=-1)
#         return x_rot

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply rotary positional embeddings to the input tensor.

#         Parameters
#         ----------
#         x: torch.Tensor
#             The input tensor. Expected shape is (batch_size, seq_len, dim).

#         Returns
#         -------
#         torch.Tensor
#             The input tensor with rotary positional embeddings applied. The shape is (batch_size, seq_len, dim).
#         """
#         return self.apply_rotary_embeddings(x)
