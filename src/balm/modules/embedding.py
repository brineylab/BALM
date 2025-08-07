# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["RotaryPositionalEmbedding"]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embeddings, as initially described in the
    `RoFormer: Enhanced Transformer with Rotary Position Embeddings`_ paper.

    Code is based on the `HuggingFace ESM implementation`_.

    Parameters
    ----------
    dim: int
        The embedding dimension.
    References
    ----------
    .. _RoFormer: Enhanced Transformer with Rotary Position Embeddings:
        https://arxiv.org/abs/2104.09864
    .. _HuggingFace ESM implementation:
        https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/esm/modeling_esm.py#L83
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached).to(
                dtype=q.dtype
            ),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached).to(
                dtype=k.dtype
            ),
        )
