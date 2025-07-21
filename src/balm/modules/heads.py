# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .activation import get_activation_fn
from .attention import SelfAttention

__all__ = [
    "BalmLMHead",
    "BalmSequenceClassificationHead",
    "BalmAttentionSequenceClassificationHead",
]


class BalmLMHead(nn.Module):
    """
    Head for masked language modeling.

    Parameters
    ----------
    config: PretrainedConfig
        Model config.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation_fn(config.mlm_activation)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # output
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        BalmLMHead forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Features tensor of shape (batch_size, sequence_length, hidden_size).

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_dim).
        """

        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias  # proj to vocab size
        return x


class BalmSequenceClassificationHead(nn.Module):
    """
    Head for sequence-level classification tasks.

    Parameters
    ----------
    config: PretrainedConfig
        Model config.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # FFN
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation_fn(config.classifier_activation)
        self.dropout = nn.Dropout(config.hidden_dropout)

        # output
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        BalmSequenceClassificationHead forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Features tensor of shape (batch_size, sequence_length, hidden_size).

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (batch_size, num_labels).
        """

        x = features[:, 0, :]  # BOS token is the sequence representative
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)  # proj to num_labels
        return x


class BalmAttentionSequenceClassificationHead(nn.Module):
    """
    Head for sequence-level classification tasks. Attention layer is added for
    interpretability, as implemented in `this paper`_.

    .. _this paper:
        https://doi.org/10.1016/j.immuni.2024.07.022

    Parameters
    ----------
    config: PretrainedConfig
        Model config.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # attention & pooling
        self.attention = SelfAttention(
            model_dim=config.hidden_size,
            num_heads=config.classifier_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type=config.position_embedding_type,
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # FFN
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation_fn(config.classifier_activation)
        self.dropout = nn.Dropout(config.hidden_dropout)

        # output
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        BalmAttentionSequenceClassificationHead forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Features tensor of shape (batch_size, sequence_length, hidden_size).

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (batch_size, num_labels).
        """

        # attention
        residual = features
        attn_out = self.attention(
            features, padding_mask=padding_mask, need_weights=need_weights
        )
        x, attn_vals = attn_out if need_weights else (attn_out[0], None)
        x = residual + self.attn_dropout(x)
        x = self.layer_norm(x)

        # avg pooling across sequence length
        x = x.mean(dim=1)
        x = torch.flatten(x, start_dim=1)

        # FFN
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.out_proj(x)  # proj to num_labels
        return (x, attn_vals) if need_weights else x
