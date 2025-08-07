# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import math
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

        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads

        # projections
        self.in_proj = nn.Linear(model_dim, 3 * model_dim, bias=True)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=True)

        # embeddings
        self.rotary_embed = (
            RotaryPositionalEmbedding(self.head_dim)
            if position_embedding_type == "rotary"
            else None
        )

        # attention
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the self-attention layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).
        attention_mask : Optional[torch.Tensor], default=None
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
        batch_size, seq_len, _ = x.shape

        # project to query/key/value
        q, k, v = self._in_proj(x)

        # reshape -> B, S, H, D
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # reshape attention mask -> B, 1, 1, S
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask[:, None, None, :]

        # rotary embeddings
        # don't scale q manually -> F.scaled_dot_product_attention handles scaling
        if self.rotary_embed is not None:
            q, k = self.rotary_embed(q, k)

        # attention
        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout,
        )

        # reshape & project
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.model_dim)
        )
        attn_out = self.out_proj(attn_out)

        # optionally compute attention weights
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            return attn_out, attn_weights

        return attn_out, None

    def _in_proj(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract query/key/value projections using in_proj linear layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).
        """
        qkv = self.in_proj(x)
        return qkv.chunk(3, dim=-1)
