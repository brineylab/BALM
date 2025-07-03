# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import RotaryPositionalEmbedding

__all__ = ["SelfAttention"]


class SelfAttention(nn.Module):
    """
    Self-attention block with optional rotary positional embeddings.

    Parameters:
    -----------
    model_dim: int
        Model dimension.
    num_heads: int
        Number of attention heads.
    dropout: float, default=0.1
        Dropout rate.
    position_embedding_type: str, default="rotary"
        Position embedding type. Only used if rotary embeddings are specified.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"Model dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.all_head_size = self.num_heads * self.head_dim

        # embeddings
        self.rotary_embed = (
            RotaryPositionalEmbedding(self.head_dim)
            if position_embedding_type == "rotary"
            else None
        )

        # attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the self-attention layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).
        padding_mask : Optional[torch.Tensor], default=None
            Padding mask of shape (batch_size, seq_len).
        need_weights : bool, default=False
            Whether to return attention weights.

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim).
        attn_weights : Optional[torch.Tensor], default=None
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
            Only returned if `need_weights` is True.
        """

        # project to query/key/value
        q, k, v = self._in_proj(x)

        # rotary embeddings
        if self.rotary_embed is not None:
            batch_size, seq_len, _ = x.shape
            positions = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            )

            # reshape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)

            # apply rotary embeddings
            q = self.rotary_embed(q, positions)
            k = self.rotary_embed(k, positions)

            # reshape back to (batch_size, seq_len, model_dim)
            q = q.view(batch_size, seq_len, -1)
            k = k.view(batch_size, seq_len, -1)

        # attention
        attn_out = self.self_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=padding_mask,
            need_weights=need_weights,
        )

        return attn_out

    def _in_proj(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract query/key/value projections using attention's parameters.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).
        """
        # use attention's combined projection weights/biases
        combined = F.linear(
            x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        )
        return torch.chunk(combined, 3, dim=-1)
