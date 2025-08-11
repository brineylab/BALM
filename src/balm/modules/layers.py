# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .attention import SelfAttention
from .ffn import DenseFFN, GluFFN, SparseFFN

__all__ = ["DenseTransformerLayer", "SparseTransformerLayer"]


class DenseTransformerLayer(nn.Module):
    """
    Standard (dense) transformer layer

    Parameters:
    -----------
    config: PretrainedConfig
        Model config.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # attention
        self.attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = SelfAttention(
            model_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type=config.position_embedding_type,
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # FFN
        self.ffn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        ffn_class = GluFFN if "glu" in config.activation else DenseFFN
        self.ffn = ffn_class(
            model_dim=config.hidden_size,
            ffn_dim=config.intermediate_size,
            activation=config.activation,
            bias=config.ffn_bias,
        )
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the DenseTransformerLayer.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim)
        attention_mask: Optional[torch.Tensor], default=None
            Boolean mask indicating padded positions (batch_size, seq_len)
        need_weights: bool, default=False
            Whether to return attention values.

            .. warning::
                If ``need_weights`` is ``True``, weights are recomputed
                and may differ slightly from the internal SDPA values

        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim)
        """

        # attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_out, attn_vals = self.attention(
            x,
            attention_mask=attention_mask,
            need_weights=need_weights,
        )
        x = residual + self.attn_dropout(attn_out)

        # FFN
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + self.ffn_dropout(x)

        return (x, attn_vals) if need_weights else x


class SparseTransformerLayer(nn.Module):
    """
    Sparse Transformer layer.

    Parameters:
    -----------
    config: PretrainedConfig
        Model config.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # attention
        self.attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = SelfAttention(
            model_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type=config.position_embedding_type,
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # sparse FFN
        self.ffn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.sparse_ffn = SparseFFN(
            model_dim=config.hidden_size,
            expert_ffn_dims=config.expert_intermediate_size,
            shared_ffn_dim=config.shared_expert_intermediate_size,
            expert_activation=config.expert_activation,
            expert_bias=config.expert_bias,
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            expert_capacity_type=config.expert_capacity_type,
            expert_capacity=config.expert_capacity,
            k=config.num_experts_per_tok,
            top_p_threshold=config.top_p_threshold,
            router_type=config.router_type,
            router_bias=config.router_bias,
            router_dtype=config.router_dtype,
            router_jitter=config.router_jitter,
        )
        self.ffn_dropout = nn.Dropout(config.expert_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the SparseTransformerLayer.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
        attention_mask: Optional[torch.Tensor], default=None
            Boolean mask indicating padded positions (batch_size, seq_len)
        need_weights: bool, default=False
            Whether to return attention values.

            .. warning::
                If ``need_weights`` is ``True``, weights are recomputed
                and may differ slightly from the internal SDPA values

        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
        """

        # attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_out, attn_vals = self.attention(
            x,
            attention_mask=attention_mask,
            need_weights=need_weights,
        )
        x = residual + self.attn_dropout(attn_out)

        # sparse FFN
        residual = x
        x = self.ffn_layer_norm(x)
        ffn_out, router_tuple = self.sparse_ffn(x)
        x = residual + self.ffn_dropout(ffn_out)

        return (x, attn_vals, router_tuple) if need_weights else (x, router_tuple)
