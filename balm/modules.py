# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from .activation import get_activation_fn
from .embedding import RelativePositionalEmbedding, RotaryPositionalEmbedding
from .router import TopKRouter

__all__ = [
    # layers
    "DenseTransformerLayer",
    "SparseTransformerLayer",
    "HybridSparseTransformerLayer",
    "SparseFFN",
    "Expert",
    # heads
    "BalmLMHead",
    "BalmSequenceClassificationHead",
    # # outputs
    # "MaskedLMOutput",
    # "ClassifierOutput",
]


# # =================================
# #
# #           ATTENTION
# #
# # =================================


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, config: PretrainedConfig):
#         super().__init__()
#         self.config = config
#         self.num_heads = config.num_attention_heads
#         self.head_dim = config.hidden_size // config.num_attention_heads
#         self.all_head_dim = self.num_heads * self.head_dim

#         self.query = nn.Linear(config.hidden_size, self.all_head_dim)
#         self.key = nn.Linear(config.hidden_size, self.all_head_dim)
#         self.value = nn.Linear(config.hidden_size, self.all_head_dim)

#         self.dropout = nn.Dropout(config.attention_dropout)
#         self.out_proj = nn.Linear(self.all_head_dim, config.hidden_size)

#         self.rotary = (
#             RotaryPositionalEmbedding(self.head_dim)
#             if config.position_embedding_type == "rotary"
#             else None
#         )
#         self.relative = (
#             RelativePositionalEmbedding(
#                 self.num_heads, max_position=config.max_position_embeddings
#             )
#             if config.position_embedding_type == "relative"
#             else None
#         )

#     def forward(self, hidden_states, attention_mask=None):
#         b, s, h = hidden_states.size()

#         q = self.query(hidden_states).view(b, s, self.num_heads, self.head_dim)
#         k = self.key(hidden_states).view(b, s, self.num_heads, self.head_dim)
#         v = self.value(hidden_states).view(b, s, self.num_heads, self.head_dim)

#         if self.rotary is not None:
#             q = self.rotary(q, s)
#             k = self.rotary(k, s)

#         # prepare attention_mask and relative positions
#         # scaled_dot_product_attention expects attn_mask shape: [B, num_heads, S, S]
#         if attention_mask is not None:
#             # attention_mask is typically [b, 1, 1, s], expand it
#             # we need [b, num_heads, s, s]
#             attention_mask = attention_mask.expand(b, self.num_heads, s, s)

#         if self.relative is not None:
#             rel_pos_bias = self.relative(s)  # [s, s, h]
#             rel_pos_bias = rel_pos_bias.permute(2, 0, 1).unsqueeze(0)  # [1, h, s, s]
#             rel_pos_bias = rel_pos_bias.expand(b, -1, s, s)  # [b, h, s, s]
#             # Add relative bias to attention_mask (both are additive)
#             if attention_mask is None:
#                 attention_mask = rel_pos_bias
#             else:
#                 attention_mask = attention_mask + rel_pos_bias

#         # transpose to [b, h, s, d] for scaled_dot_product_attention
#         q = q.transpose(1, 2)  # [b, h, s, d]
#         k = k.transpose(1, 2)  # [b, h, s, d]
#         v = v.transpose(1, 2)  # [b, h, s, d]

#         # use scaled_dot_product_attention (flash attention if available, else SDPA)
#         # torch will automatically use fast paths if conditions are met.
#         attn_output = F.scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             attn_mask=attention_mask,  # additive mask
#             dropout_p=self.dropout.p,
#             is_causal=False,
#         )  # attn_output: [b, h, s, d]

#         # reshape back
#         attn_output = attn_output.transpose(1, 2).contiguous().view(b, s, h)
#         out = self.out_proj(attn_output)
#         out = self.dropout(out)
#         return out


# =================================
#
#             HEADS
#
# =================================


class BalmLMHead(nn.Module):
    """
    Head for masked language modeling.

    Parameters
    ----------
    hidden_size : int
        Hidden size.

    output_dim : int
        Output dimension.

    activation : str, optional
        Activation function to use. The default is "gelu".

    """

    def __init__(self, hidden_size: int, output_dim: int, activation: str = "gelu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = get_activation_fn(activation)

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
        x = self.decoder(x) + self.bias
        return x


class BalmSequenceClassificationHead(nn.Module):
    """
    Head for sequence-level classification tasks.

    Parameters
    ----------
    hidden_size : int
        Hidden size.

    num_labels : int
        Number of labels.

    dropout : float, optional
        Dropout rate. The default is 0.0.

    activation : str, optional
        Activation function to use. The default is "tanh".

    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.0,
        activation: str = "tanh",
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

        # activation
        self.activation = get_activation_fn(activation)

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
        x = self.out_proj(x)
        return x


class BalmTokenClassificationHead(nn.Module):
    """
    Head for token-level classification tasks.

    Parameters
        ----------
        hidden_size : int
            Hidden size.

        num_labels : int
            Number of labels.

        dropout : float, optional
            Dropout rate. The default is 0.0.

    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        BalmTokenClassificationHead forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Features tensor of shape (batch_size, sequence_length, hidden_size).

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, num_labels).

        """
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


# =================================
#
#            EXPERTS
#
# =================================


class Expert(nn.Module):
    """
    Expert module for a Sparse Transformer layer.

    Parameters:
    -----------
    model_dim : int
        Model dimension.

    ffn_dim : int
        Feed-forward network dimension.

    dropout : float, optional
        Dropout rate. The default is 0.0.

    activation : str, optional
        Activation function to use. The default is "swiglu".

    bias : bool, optional
        Whether to use bias. The default is True.

    Returns
    -------
    x : torch.Tensor
        Output tensor of shape (batch_size, sequence_length, model_dim).

    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = "swiglu",
        bias: bool = True,
    ):
        super().__init__()
        self.wi = nn.Linear(
            model_dim,
            ffn_dim,
            bias=bias,
        )
        self.wo = nn.Linear(
            ffn_dim,
            model_dim,
            bias=bias,
        )
        self.activation = get_activation_fn(activation, dim=ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input hidden states.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        x = self.wi(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.wo(x)
        return x


# class Expert(nn.Module):
#     """
#     Expert module for a Sparse Transformer layer.

#     Parameters:
#     -----------
#     config : PretrainedConfig
#         Configuration object.

#     """

#     def __init__(
#         self,
#         config: PretrainedConfig,
#     ):
#         super().__init__()
#         self.wi = nn.Linear(
#             config.hidden_size,
#             config.intermediate_size,
#             bias=config.expert_bias,
#         )
#         self.wo = nn.Linear(
#             config.intermediate_size,
#             config.hidden_size,
#             bias=config.expert_bias,
#         )
#         self.activation = get_activation_fn(
#             config.expert_activation, dim=config.hidden_size
#         )
#         self.dropout = nn.Dropout(config.expert_dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         x : torch.Tensor
#             Output tensor of shape (batch_size, sequence_length, embed_dim).
#         """
#         x = self.wi(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.wo(x)
#         return x


# class Expert(nn.Module):
#     """
#     Expert module for a Sparse Transformer layer.

#     Parameters:
#     -----------
#     embed_dim : int
#         Embedding dimension.

#     ffn_embed_dim : int
#         Feed-forward network embedding dimension. Typically 4x the embedding dimension.

#     dropout_rate : float
#         Dropout rate. The default is ``0.0``.

#     activation : str, optional
#         Activation function to use. One of "swiglu", "relu", or "gelu". The default is "gelu".
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         dropout: float = 0.0,
#         activation: str = "swiglu",
#     ):
#         super().__init__()
#         out_ffn_dim = ffn_dim
#         if activation.lower() == "swiglu":
#             out_ffn_dim = ffn_dim // 2
#             self.activation = SwiGLU()
#         elif activation.lower() == "gelu":
#             self.activation = nn.GELU()
#         else:
#             self.activation = nn.ReLU()
#         self.wi = nn.Linear(embed_dim, ffn_dim, bias=False)
#         self.wo = nn.Linear(out_ffn_dim, embed_dim, bias=False)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         x : torch.Tensor
#             Output tensor of shape (batch_size, sequence_length, embed_dim).
#         """
#         x = self.wi(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         if (
#             isinstance(self.wo.weight, torch.Tensor)
#             and x.dtype != self.wo.weight.dtype
#             and self.wo.weight.dtype != torch.int8
#         ):
#             x = x.to(self.wo.weight.dtype)
#         x = self.wo(x)
#         return x


# =================================
#
#             FFNs
#
# =================================


class DenseFFN(nn.Module):
    """
    Standard (dense) feed-forward network.

    Parameters:
    -----------
    model_dim: int
        Token embedding dimension.

    ffn_dim: int, default=None
        Feed-forward network dimension. If not provided, it will be set to 4x the model dimension.

    activation: str, default="swiglu"
        Activation function to use.

    bias: bool, default=True
        Whether to use bias.

    dropout: float, default=0.0
        Dropout rate.

    Input shape: (batch_size, seq_len, model_dim)
    Output shape: (batch_size, seq_len, model_dim)

    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int = None,
        activation: str = "swiglu",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.wi = nn.Linear(model_dim, ffn_dim, bias=bias)
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias)
        self.activation = get_activation_fn(activation, dim=ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DenseFFN layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim).

        """
        x = self.wi(x)
        x = self.activation(x)
        return self.dropout(self.wo(x))


class SwigluFFN(nn.Module):
    """
    SwiGLU-activated feed-forward network.
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int = None,
        bias: bool = True,
        dropout: float = 0.0,
        activation: str = "swiglu",  # to make init signature compatible with DenseFFN
    ):
        super().__init__()
        self.gate_linear = nn.Linear(model_dim, ffn_dim, bias=bias)
        self.value_linear = nn.Linear(model_dim, ffn_dim, bias=bias)
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_linear(x)
        value = self.value_linear(x)
        x = value * F.silu(gate)
        return self.dropout(self.wo(x))


class SparseFFN(nn.Module):
    """Sparse Mixture of Experts layer with capacity constraints.

    Parameters:
    -----------
    model_dim: int
        Token embedding dimension.

    ffn_dim: int, default=None
        Feed-forward network dimension. If not provided, it will be set to 4x the model dimension.

    activation: str, default="swiglu"
        Activation function to use.

    bias: bool, default=True
        Whether to use bias.

    dropout: float, default=0.0
        Dropout rate.

    num_experts: int
        Number of experts.

    max_capacity: Union[int, float], default=1.0
        Expert capacity (int = absolute, float = multiplier).

    k: int, default=1
        Number of experts per token.

    Input shape: (batch_size, seq_len, model_dim)
    Output shape: (batch_size, seq_len, model_dim)

    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_experts: int,
        max_capacity: Union[int, float] = 1.0,
        k: int = 1,
        activation: str = "swiglu",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # router
        self.router = TopKRouter(model_dim, num_experts)

        # experts
        expert_class = SwigluFFN if activation.lower() == "swiglu" else DenseFFN
        self.experts = nn.ModuleList(
            [
                expert_class(
                    model_dim=model_dim,
                    ffn_dim=ffn_dim,
                    activation=activation,
                    bias=bias,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )
        self.num_experts = num_experts
        self.k = k

        # capacity
        if isinstance(max_capacity, float):
            self.capacity_multiplier = max_capacity
            self.absolute_capacity = None
        else:
            self.absolute_capacity = max_capacity
            self.capacity_multiplier = None

    def _compute_capacity(self, num_tokens: int) -> int:
        """
        Determine expert capacity

        Parameters:
        -----------
        num_tokens: int
            Number of tokens in the batch.

        Returns:
        --------
        capacity: int
            Expert capacity.

        """
        if self.capacity_multiplier:
            return int(self.capacity_multiplier * num_tokens)
        return self.absolute_capacity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SparseFFN layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim).

        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.view(-1, d_model)  # (num_tokens, d_model)

        # get routing information
        logits, probs, indices = self.router(x_flat, self.k)
        capacity = self._compute_capacity(num_tokens)

        output = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            # find tokens that selected this expert in their top-k
            mask = (indices == expert_idx).any(dim=1)
            if not mask.any():
                continue
            # get candidate tokens and their scores
            candidate_indices = torch.nonzero(mask).squeeze(-1)
            expert_scores = logits[candidate_indices, expert_idx]
            # sort by descending expert affinity
            sorted_scores, score_order = torch.sort(expert_scores, descending=True)
            sorted_candidates = candidate_indices[score_order]
            # apply capacity constraint
            if sorted_candidates.numel() > capacity:
                sorted_candidates = sorted_candidates[:capacity]
            if sorted_candidates.numel() == 0:
                continue
            # get routing weights for selected tokens
            expert_input = x_flat[sorted_candidates]
            expert_positions = (indices[sorted_candidates] == expert_idx).nonzero(
                as_tuple=True
            )[1]
            weights = (
                probs[sorted_candidates]
                .gather(1, expert_positions.unsqueeze(1))
                .squeeze(1)
            )
            # compute and accumulate expert contribution
            expert_output = expert(expert_input) * weights.unsqueeze(1)
            output[sorted_candidates] += expert_output

        output = output.view(batch_size, seq_len, d_model)
        return output, (logits, indices)


# class SparseMLP(nn.Module):
#     """
#     Sparse MLP layer, consisting of a router and a set of experts.

#     Parameters:
#     -----------
#     config : PretrainedConfig
#         Configuration object.

#     """

#     def __init__(self, config: PretrainedConfig):
#         super().__init__()
#         self.config = config
#         self.num_experts = config.num_experts
#         self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
#         if config.router_type == "expert choice":
#             self.router = ExpertChoiceRouter(config)
#         else:
#             self.router = TopKRouter(config)

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
#         """
#         Sparse MLP forward pass.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         output : Tuple[torch.Tensor, Tuple]
#             A tuple containing the following:
#             - x : torch.Tensor
#                 Output tensor of shape (batch_size, sequence_length, embed_dim).
#             - router_outputs : Tuple[torch.Tensor, torch.Tensor]
#                 A tuple containing the following:
#                     - router_logits : torch.Tensor
#                         Router logits of shape (batch_size, sequence_length, num_experts).
#                     - expert_mask : torch.Tensor
#                         Expert mask of shape (batch_size, sequence_length, num_experts).

#         """
#         # router
#         expert_mask, router_probs, router_logits = self.router(x)
#         expert_outputs = []

#         # experts
#         for idx, expert in enumerate(self.experts):
#             token_indices = expert_mask[..., idx].bool()
#             expert_output = expert(x[token_indices]).to(x.dtype)
#             expanded_output = torch.zeros_like(x)
#             expanded_output[token_indices] = expert_output
#             expert_outputs.append(expanded_output)

#         # combine outputs from the selected tokens for each expert
#         x = torch.stack(expert_outputs, dim=-1) * expert_mask.unsqueeze(-2)
#         # multiply by router probs before summing
#         x = torch.sum(x * router_probs.unsqueeze(-2), dim=-1)

#         return x, (router_logits, expert_mask)


# class SparseLayer_R1(nn.Module):
#     """Sparse Mixture of Experts layer with expert capacity constraints.

#     Parameters
#     ----------
#     d_model : int
#         Token embedding dimension

#     num_experts : int
#         Number of parallel experts

#     max_capacity : Union[int, float], optional
#         Expert capacity (int = absolute, float = multiplier)
#         The default is 1.0.

#     k : int, optional
#         Number of experts per token (default 1)

#     Input shape: (batch_size, seq_len, d_model)
#     Output shape: (batch_size, seq_len, d_model)

#     """

#     def __init__(
#         self,
#         d_model: int,
#         num_experts: int,
#         max_capacity: Union[int, float] = 1.0,
#         k: int = 1,
#     ):
#         super().__init__()
#         self.router = Router_R1(d_model, num_experts)
#         self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
#         self.num_experts = num_experts
#         self.k = k

#         if isinstance(max_capacity, float):
#             self.capacity_multiplier = max_capacity
#             self.absolute_capacity = None
#         else:
#             self.absolute_capacity = max_capacity
#             self.capacity_multiplier = None

#     def _compute_capacity(self, num_tokens: int) -> int:
#         """Determine expert capacity based on configuration."""
#         if self.capacity_multiplier:
#             return int(self.capacity_multiplier * num_tokens)
#         return self.absolute_capacity

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, d_model = x.shape
#         num_tokens = batch_size * seq_len
#         x_flat = x.view(-1, d_model)  # (num_tokens, d_model)

#         # Get routing information
#         logits, probs, indices = self.router(x_flat, self.k)
#         capacity = self._compute_capacity(num_tokens)

#         # Initialize output and expert usage tracking
#         output = torch.zeros_like(x_flat)

#         # Process each expert independently
#         for expert_idx, expert in enumerate(self.experts):
#             # Find tokens that selected this expert in their top-k
#             mask = (indices == expert_idx).any(dim=1)
#             if not mask.any():
#                 continue

#             # Get candidate tokens and their scores
#             candidate_indices = torch.nonzero(mask).squeeze(-1)
#             expert_scores = logits[candidate_indices, expert_idx]

#             # Sort by descending expert affinity
#             sorted_scores, score_order = torch.sort(expert_scores, descending=True)
#             sorted_candidates = candidate_indices[score_order]

#             # Apply capacity constraint
#             if sorted_candidates.numel() > capacity:
#                 sorted_candidates = sorted_candidates[:capacity]

#             # Skip empty expert batches
#             if sorted_candidates.numel() == 0:
#                 continue

#             # Get routing weights for selected tokens
#             expert_input = x_flat[sorted_candidates]
#             expert_positions = (indices[sorted_candidates] == expert_idx).nonzero(
#                 as_tuple=True
#             )[1]
#             weights = (
#                 probs[sorted_candidates]
#                 .gather(1, expert_positions.unsqueeze(1))
#                 .squeeze(1)
#             )

#             # Compute and accumulate expert contribution
#             expert_output = expert(expert_input) * weights.unsqueeze(1)
#             output[sorted_candidates] += expert_output

#         return output.view(batch_size, seq_len, d_model)


# b, s, h = hidden_states.size()  # [batch_size, sequence_length, embed_dim]
# if self.config.expert_capacity_type == "absolute":
#     capacity = int(self.config.expert_capacity)
# else:
#     capacity = int(self.config.expert_capacity * s / self.config.num_experts)

# dispatch_mask, combine_weights, aux_loss, z_loss = self.router(
#     hidden_states, seq_len=s, capacity=capacity
# )

# expert_outputs = []
# for i, expert in enumerate(self.experts):
#     expert_mask = dispatch_mask[..., i]  # [b, s]
#     tokens_for_expert = hidden_states[expert_mask]  # [num_tokens_for_expert, h]
#     if tokens_for_expert.size(0) > 0:
#         out = expert(tokens_for_expert)
#     else:
#         out = torch.zeros_like(tokens_for_expert)
#     expert_outputs.append(out)

# combined_output = torch.zeros_like(hidden_states)
# for i, expert_out in enumerate(expert_outputs):
#     expert_mask = dispatch_mask[..., i]
#     cw = combine_weights[expert_mask, i].unsqueeze(-1)
#     combined_output[expert_mask] += expert_out * cw

# # combined output -> [batch_size, sequence_length, embed_dim]
# return combined_output, aux_loss, z_loss


# class SparseMLP(nn.Module):
#     """
#     Implementation of a Sparse MLP module, for use in Mixture-of-Experts models.

#     Parameters:
#     -----------
#     embed_dim : int
#         Embedding dimension.

#     ffn_dim : int
#         Feedforward dimension.

#     num_experts : int
#         Number of experts.

#     expert_capacity : int
#         Capacity of each expert.

#     top_k : int, optional
#         Top k for the router. The default is 1.

#     activation : str, optional
#         Activation function to use. The default is "swiglu".

#     expert_ffn_dropout : float, optional
#         Dropout rate for the expert feedforward layer. The default is 0.0.

#     router_dtype : str, optional
#         Dtype for the router. The default is "float32".

#     router_bias : bool, optional
#         Whether to use bias for the router. The default is False.

#     router_jitter : float, optional
#         Jitter for the router. The default is 0.0.

#     router_ignore_padding_tokens : bool, optional
#         Whether to ignore padding tokens for the router. The default is True.

#     router_class : nn.Module, optional
#         Router class to use. The default is ``TopKRouter``.

#     expert_class : nn.Module, optional
#         Expert class to use. The default is ``Expert``.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_experts: int,
#         expert_capacity: int,
#         num_shared_experts: int = 0,
#         send_bos_to_all_experts: bool = True,
#         top_k: int = 1,
#         activation: str = "swiglu",
#         expert_ffn_dropout: float = 0.0,
#         router_dtype: str = "float32",
#         router_bias: bool = False,
#         router_jitter: float = 0.0,
#         router_ignore_padding_tokens: bool = True,
#         router_class: nn.Module = TopKRouter,
#         expert_class: nn.Module = Expert,
#     ):
#         super().__init__()
#         self.router = router_class(
#             embed_dim=embed_dim,
#             num_experts=num_experts,
#             expert_capacity=expert_capacity,
#             top_k=top_k,
#             num_shared_experts=num_shared_experts,
#             send_bos_to_all_experts=send_bos_to_all_experts,
#             dtype=router_dtype,
#             bias=router_bias,
#             jitter=router_jitter,
#             ignore_padding_tokens=router_ignore_padding_tokens,
#         )
#         self.experts = nn.ModuleDict()
#         for idx in range(num_experts):
#             self.experts[f"expert_{idx}"] = expert_class(
#                 embed_dim=embed_dim,
#                 ffn_dim=ffn_dim,
#                 dropout=expert_ffn_dropout,
#                 activation=activation,
#             )

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
#         """
#         Route tokens to experts and process them.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         output : Tuple[torch.Tensor, Tuple]
#             A tuple containing the following:
#              - x : torch.Tensor
#                 Output tensor of shape (batch_size, sequence_length, embed_dim).
#              - router_outputs : Tuple[torch.Tensor, torch.Tensor]
#                 A tuple containing the following:
#                  - router_logits : torch.Tensor
#                     Router logits of shape (batch_size, sequence_length, num_experts).
#                  - expert_mask : torch.Tensor
#                     Expert mask of shape (batch_size, sequence_length, num_experts).
#         """
#         # router
#         expert_mask, router_probs, router_logits = self.router(x)
#         expert_outputs = []

#         # experts
#         for idx, expert in self.experts.items():
#             int_idx = int(idx.split("_")[-1])
#             token_indices = expert_mask[..., int_idx].bool()
#             expert_output = expert(x[token_indices]).to(x.dtype)
#             expanded_output = torch.zeros_like(x)
#             expanded_output[token_indices] = expert_output
#             expert_outputs.append(expanded_output)

#         # combine outputs from the selected tokens for each expert
#         x = torch.stack(expert_outputs, dim=-1) * expert_mask.unsqueeze(-2)
#         # multiply by router probs before summing
#         x = torch.sum(x * router_probs.unsqueeze(-2), dim=-1)

#         return x, (router_logits, expert_mask)


# =================================
#
#           ATTENTION
#
# =================================


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
        Position embedding type. Only used if rotary embeddings are used (i.e. `position_embedding_type="rotary"`).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()
        # embeddings
        if position_embedding_type == "rotary":
            self.rotary_embed = RotaryPositionalEmbedding(dim=model_dim // num_heads)
        else:
            self.rotary_embed = None

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

        """
        # project to query/key/value
        q, k, v = self._in_proj(x)

        # rotary embeddings
        if self.rotary_embed is not None:
            batch_size, seq_len, _ = x.shape
            positions = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            )
            # apply rotary embeddings
            num_heads = self.self_attn.num_heads
            head_dim = q.size(-1) // num_heads
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_heads, head_dim)
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
        # Use attention's combined projection weights/biases
        combined = F.linear(
            x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        )
        return torch.chunk(combined, 3, dim=-1)


# =================================
#
#       TRANSFORMER LAYERS
#
# =================================


class DenseTransformerLayer(nn.Module):
    """
    Standard (dense) transformer layer

    Parameters:
    -----------
    model_dim: int
        Model dimension.

    ffn_dim: int | None, default=None
        Feed-forward dimension. If not provided, it will be set to 4x the model dimension.

    num_heads: int, default=20
        Number of attention heads.

    activation: str, default="swiglu"
        Activation function to use.

    bias: bool, default=True
        Whether to use bias.

    dropout: float, default=0.1
        Dropout rate.

    position_embedding_type: str, default="rotary"
        Position embedding type. Only used if rotary embeddings are used (i.e. `position_embedding_type="rotary"`).

    Input shape: (batch_size, seq_len, model_dim)
    Output shape: (batch_size, seq_len, model_dim)

    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int | None = None,
        num_heads: int = 20,
        activation: str = "swiglu",
        bias: bool = True,
        dropout: float = 0.1,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()

        # attention
        self.self_attn = SelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            position_embedding_type=position_embedding_type,
        )

        # FFN
        ffn_class = SwigluFFN if activation.lower() == "swiglu" else DenseFFN
        self.ffn = ffn_class(
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            activation=activation,
            bias=bias,
        )

        # norm
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        # dropout
        self.dropout1 = nn.Dropout(dropout)  # attention dropout
        self.dropout2 = nn.Dropout(dropout)  # ffn dropout

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        # invert padding mask and convert to boolean, since the 🤗 DataCollatorForLanguageModeling
        # uses 0 for padding tokens and 1 for other tokens, but we want True for padding tokens and
        # False for other tokens
        if padding_mask is not None:
            padding_mask = 1 - padding_mask
            padding_mask = padding_mask.bool()

        # pre-norm
        residual = x
        x = self.norm1(x)

        # attention
        # NOTE: if need_weights is True, torch can't use optimized SDPA
        # see -> https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        attn_out = self.self_attn(
            x,
            padding_mask=padding_mask,
            need_weights=need_weights,
        )
        if need_weights:
            attn_out, attn_vals = attn_out
        else:
            attn_out = attn_out[0]
        x = residual + self.dropout1(attn_out)

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))

        if need_weights:
            return (x, attn_vals)
        return x


class SparseTransformerLayer(nn.Module):
    """Sparse Transformer Layer with Rotary Positional Embeddings and Mixture of Experts.

    Args:
        d_model: Token embedding dimension
        num_heads: Number of attention heads
        num_experts: Number of experts in SparseFFN
        max_capacity: Expert capacity (int = absolute, float = multiplier)
        k: Number of experts per token (default 1)
        dropout: Dropout probability (default 0.1)

    Input shape: (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_experts: int,
        max_capacity: Union[int, float],
        k: int = 1,
        activation: str = "swiglu",
        bias: bool = True,
        dropout: float = 0.1,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # attention
        self.self_attn = SelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            position_embedding_type=position_embedding_type,
        )

        # Sparse FFN components
        self.sparse_ffn = SparseFFN(
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            max_capacity=max_capacity,
            k=k,
            activation=activation,
            bias=bias,
        )

        # norm
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            padding_mask: Optional boolean mask indicating padded positions
                          (batch_size, seq_len)
            need_weights: Whether to return attention values, default is False

        Returns:
            torch.Tensor of shape (batch_size, seq_len, d_model)
        """
        # invert padding mask and convert to boolean, since the 🤗 DataCollatorForLanguageModeling
        # uses 0 for padding tokens and 1 for other tokens, but we want True for padding tokens and
        # False for other tokens
        if padding_mask is not None:
            padding_mask = 1 - padding_mask
            padding_mask = padding_mask.bool()

        # pre-norm
        residual = x
        x = self.norm1(x)

        # attention
        # NOTE: if need_weights is True, torch can't use optimized SDPA
        # see -> https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        attn_out = self.self_attn(
            x,
            padding_mask=padding_mask,
            need_weights=need_weights,
        )
        if need_weights:
            attn_out, attn_vals = attn_out
        else:
            attn_out = attn_out[0]
        x = residual + self.dropout1(attn_out)

        # sparse FFN
        residual = x
        x = self.norm2(x)
        ffn_out, router_tuple = self.sparse_ffn(x)
        x = residual + self.dropout2(ffn_out)

        if need_weights:
            return (x, attn_vals, router_tuple)
        return (x, router_tuple)


# class DenseTransformerLayer(nn.Module):
#     """
#     Dense transformer layer.

#     Parameters:
#     -----------
#     config : PretrainedConfig
#         Model configuration class with all the parameters of the model.

#     """

#     def __init__(self, config: PretrainedConfig):
#         super().__init__()
#         self.config = config

#         # embeddings
#         if config.position_embedding_type == "rotary":
#             self.rotary_embeddings = RotaryPositionalEmbedding(config.hidden_size)

#         # norm
#         self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#         # attention
#         # NOTE: if config.output_attentions is True, torch can't use optimized SDPA
#         # see -> https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
#         self.attention = nn.MultiheadAttention(
#             embed_dim=config.hidden_size,
#             num_heads=config.num_attention_heads,
#             dropout=config.attention_dropout,
#             batch_first=True,
#         )

#         # feedforward
#         # factor = 2 if config.expert_activation == "swiglu" else 1
#         # self.activation = get_activation_fn(config.activation)
#         # self.feed_forward = nn.Sequential(
#         #     nn.Linear(config.hidden_size, config.intermediate_size * factor),
#         #     self.activation,
#         #     nn.Dropout(config.hidden_dropout),
#         #     nn.Linear(config.intermediate_size, config.hidden_size),
#         # )
#         factor = 2 if config.activation.lower() == "swiglu" else 1
#         self.ffn_in = nn.Linear(config.hidden_size, config.intermediate_size * factor)
#         self.ffn_out = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.ffn_activation = get_activation_fn(config.activation)

#         # dropout
#         self.dropout = nn.Dropout(config.dropout)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#     ):
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, hidden_size).

#         attention_mask : torch.Tensor, optional
#             Attention mask of shape (batch_size, sequence_length). The default is None.

#         need_weights : bool, optional
#             Whether to return the attention weights. The default is False.

#         Returns:
#         --------
#         output : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
#             If ``config.output_attentions`` is ``True``, the output will be a tuple of
#             (x, attn). Otherwise, the output will be just x.

#             x : torch.Tensor
#                 Output tensor of shape (batch_size, sequence_length, hidden_size).
#             attn : torch.Tensor
#                 Attention weights of shape (batch_size, num_heads, sequence_length, sequence_length).

#         """

#         # invert attention mask and convert to boolean, since the 🤗 DataCollatorForLanguageModeling
#         # uses 0 for padding tokens and 1 for other tokens, but we want True for padding tokens and
#         # False for other tokens
#         if attention_mask is not None:
#             attention_mask = 1 - attention_mask
#             attention_mask = attention_mask.bool()

#         # pre-norm
#         residual = x
#         if self.config.pre_norm:
#             x = self.norm1(x)

#         # positional embeddings
#         if self.config.position_embedding_type == "rotary":
#             k = self.rotary_embeddings(x)
#             q = self.rotary_embeddings(x)
#             v = x
#         else:
#             k, q, v = x, x, x

#         # attention
#         x = self.attention(
#             k,
#             q,
#             v,
#             key_padding_mask=attention_mask,
#             need_weights=need_weights,
#         )
#         if need_weights:
#             x, attn = x
#         else:
#             x = x[0]
#         x = residual + self.dropout(x)

#         # post-norm
#         if not self.config.pre_norm:
#             x = self.norm1(x)

#         # pre-norm
#         residual = x
#         if self.config.pre_norm:
#             x = self.norm2(x)

#         # feedforward
#         # x = self.feed_forward(x)
#         x = self.ffn_in(x)
#         x = self.ffn_activation(x)
#         x = self.dropout(x)
#         x = self.ffn_out(x)
#         x = residual + x

#         # post-norm
#         if not self.config.pre_norm:
#             x = self.norm2(x)

#         # outputs
#         if need_weights:
#             return x, attn
#         return x


# class DenseTransformerLayer(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_heads: int,
#         max_length: int,
#         dropout: float = 0.1,
#         attention_dropout: float = 0.0,
#         token_embedding_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#         activation: str = "swiglu",
#         positional_embedding_type: Optional[str] = "rotary",
#         pre_norm: bool = True,
#     ):
#         super().__init__()
#         self.pre_norm = pre_norm

#         # embeddings
#         if positional_embedding_type is None:
#             self.positional_embeddings = None
#         elif positional_embedding_type.lower() == "rotary":
#             self.positional_embeddings = RotaryPositionalEmbedding(
#                 embed_dim, max_length
#             )
#         elif positional_embedding_type.lower() == "relative":
#             self.positional_embeddings = RelativePositionalEmbedding(
#                 embed_dim, max_length
#             )
#         else:
#             raise ValueError(
#                 f"Invalid positional embedding type: {positional_embedding_type}. Valid options are 'rotary', 'relative', or None."
#             )

#         # norm
#         self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

#         # attention
#         self.attention = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=attention_dropout,
#             batch_first=True,
#         )

#         # activation
#         post_activation_ffn_dim = ffn_dim
#         if activation.lower() == "gelu":
#             self.activation_fn = nn.GELU()
#         elif activation.lower() == "swiglu":
#             self.activation_fn = SwiGLU()
#             post_activation_ffn_dim = ffn_dim // 2
#         elif activation.lower() == "relu":
#             self.activation_fn = nn.ReLU()
#         else:
#             raise ValueError(
#                 f"Invalid activation function: {activation}. Valid options are 'swiglu', 'gelu', or 'relu'."
#             )

#         # feedforward
#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_dim, ffn_dim),
#             self.activation_fn,
#             nn.Linear(post_activation_ffn_dim, embed_dim),
#         )

#         # dropout
#         self.dropout = nn.Dropout(dropout)
#         self.embedding_dropout = nn.Dropout(token_embedding_dropout)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#     ):
#         # pre-norm
#         residual = x
#         if self.pre_norm:
#             x = self.norm1(x)

#         # positional embeddings
#         if self.positional_embeddings is not None:
#             x = self.embedding_dropout(self.positional_embeddings(x))

#         # attention
#         x = self.attention(
#             x,
#             x,
#             x,
#             attn_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#         )
#         if need_weights:
#             x, weights = x
#         else:
#             x = x[0]
#         x = residual + self.dropout(x)

#         # post-norm
#         if not self.pre_norm:
#             x = self.norm1(x)

#         # pre-norm
#         residual = x
#         if self.pre_norm:
#             x = self.norm2(x)

#         # feedforward
#         x = self.feed_forward(x)
#         x = residual + self.dropout(x)

#         # post-norm
#         if not self.pre_norm:
#             x = self.norm2(x)

#         # outputs
#         if need_weights:
#             return x, weights
#         return x


# class SparseTransformerLayer(nn.Module):
#     """
#     Sparse transformer layer.

#     Parameters:
#     -----------
#     config : PretrainedConfig
#         Model configuration class with all the parameters of the model.

#     """

#     def __init__(
#         self,
#         config: PretrainedConfig,
#     ):
#         super().__init__()
#         self.config = config

#         # embeddings
#         if config.position_embedding_type == "rotary":
#             self.rotary_embeddings = RotaryPositionalEmbedding(config.hidden_size)

#         # norm
#         self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#         # attention
#         # NOTE: if config.output_attentions is True, torch can't use optimized SDPA
#         # see -> https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
#         self.attention = nn.MultiheadAttention(
#             embed_dim=config.hidden_size,
#             num_heads=config.num_attention_heads,
#             dropout=config.attention_dropout,
#             batch_first=True,
#         )

#         # sparse feedforward
#         self.mlp = SparseMLP(config)

#         # dropout
#         self.dropout = nn.Dropout(config.dropout)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, hidden_size).

#         attention_mask : torch.Tensor, optional
#             Attention mask of shape (batch_size, sequence_length). The default is None.

#         need_weights : bool, optional
#             Whether to return the attention weights. The default is False.

#         Returns:
#         --------
#         output : Tuple[torch.Tensor, Tuple]

#             If ``config.output_attentions`` is ``True``, output is a tuple of (x, attn, router_tuple).
#             Otherwise, output is a tuple of (x, router_tuple).

#             x: torch.Tensor
#                 Output tensor of shape (batch_size, sequence_length, hidden_size).
#             attn: torch.Tensor
#                 Attention weights of shape (batch_size, num_heads, sequence_length, sequence_length).
#             router_tuple: Tuple
#                 Tuple of router logits and expert mask. Both are of shape
#                 (batch_size, sequence_length, num_experts).

#         """
#         # invert attention mask and convert to boolean, since the 🤗 DataCollatorForLanguageModeling
#         # uses 0 for padding tokens and 1 for other tokens, but we want True for padding tokens and
#         # False for other tokens
#         if attention_mask is not None:
#             attention_mask = 1 - attention_mask
#             attention_mask = attention_mask.bool()

#         # pre-norm
#         residual = x
#         if self.config.pre_norm:
#             x = self.norm1(x)

#         # # positional embeddings
#         # x = self.embedding_dropout(self.positional_embeddings(x))

#         # # attention
#         # x = self.self_attn(
#         #     x,
#         #     x,
#         #     x,
#         #     key_padding_mask=attention_mask,
#         #     need_weights=self.config.output_attentions,
#         # )

#         # positional embeddings
#         if self.config.position_embedding_type == "rotary":
#             k = self.rotary_embeddings(x)
#             q = self.rotary_embeddings(x)
#             v = x
#         else:
#             k, q, v = x, x, x

#         # attention
#         x = self.attention(
#             k,
#             q,
#             v,
#             key_padding_mask=attention_mask,
#             need_weights=need_weights,
#         )

#         if need_weights:
#             x, attn = x
#         else:
#             x = x[0]
#         x = residual + self.dropout(x)

#         # post-norm
#         if not self.config.pre_norm:
#             x = self.norm1(x)

#         # pre-norm
#         residual = x
#         if self.config.pre_norm:
#             x = self.norm2(x)

#         # feedforward
#         x, router_tuple = self.mlp(x)
#         x = residual + self.dropout(x)

#         # post-norm
#         if not self.config.pre_norm:
#             x = self.norm2(residual + x)

#         # outputs
#         if need_weights:
#             return (x, attn, router_tuple)
#         return (x, router_tuple)


# class SparseTransformerLayer(nn.Module):
#     """
#     BALM transformer layer with Mixture of Experts. Approximately follows the ESM-2
#     implementation, but differs in a few ways:
#         - includes (optional) dropout for self-attention and feedforward layers
#         - normalize **after**, not before, the self-attention and feedforward layers
#         - we don't use rotary embeddings, which aren't (yet?) compatible with
#           torch's optimized implementation of ``nn.MultiheadAttention``

#     Parameters:
#     -----------
#     config : BalmMoEConfig
#         Model configuration class with all the parameters of the model.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_heads: int,
#         max_length: int,
#         num_experts: int,
#         expert_capacity: int,
#         num_shared_experts: int = 0,
#         send_bos_to_all_experts: bool = True,
#         top_k: int = 1,
#         dropout: float = 0.1,
#         attention_dropout: float = 0.0,
#         expert_ffn_dropout: float = 0.0,
#         token_embedding_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#         activation: str = "swiglu",
#         positional_embedding_type: str = "rotary",
#         pre_norm: bool = True,
#         router_dtype: str = "float32",
#         router_bias: bool = False,
#         router_jitter: float = 0.0,
#         router_ignore_padding_tokens: bool = True,
#         expert_choice_router: bool = False,
#     ):
#         super().__init__()
#         self.pre_norm = pre_norm

#         # embeddings
#         if positional_embedding_type.lower() == "rotary":
#             self.positional_embeddings = RotaryPositionalEmbedding(
#                 embed_dim, max_length
#             )
#         else:
#             self.positional_embeddings = RelativePositionalEmbedding(
#                 embed_dim, max_length
#             )

#         # norm
#         self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

#         # attention
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=attention_dropout,
#             batch_first=True,
#         )

#         # sparse feedforward
#         self.mlp = SparseMLP(
#             embed_dim=embed_dim,
#             ffn_dim=ffn_dim,
#             num_experts=num_experts,
#             num_shared_experts=num_shared_experts,
#             send_bos_to_all_experts=send_bos_to_all_experts,
#             top_k=top_k,
#             expert_capacity=expert_capacity,
#             activation=activation,
#             expert_ffn_dropout=expert_ffn_dropout,
#             router_dtype=router_dtype,
#             router_bias=router_bias,
#             router_jitter=router_jitter,
#             router_ignore_padding_tokens=router_ignore_padding_tokens,
#             router_class=ExpertChoiceRouter if expert_choice_router else TopKRouter,
#             expert_class=Expert,
#         )

#         # dropout
#         self.dropout = nn.Dropout(dropout)
#         self.embedding_dropout = nn.Dropout(token_embedding_dropout)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#         output_router_logits: bool = True,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         attn_mask : torch.Tensor, optional
#             Attention mask of shape (batch_size * num_heads, sequence_length, sequence_length). The default is None.

#         key_padding_mask : torch.Tensor, optional
#             Mask of shape (batch_size, sequence_length). The default is None.

#         need_weights : bool, optional
#             Whether to return attention weights. The default is False.

#             .. note::
#                 if `need_weights` is ``True``, the output will be a tuple of (x, attn). Also,
#                 nn.MultiHeadAttention will not be able to use the optimized torch implementation
#                 of ``scaled_dot_product_attention``. See `here`_ for more details.

#         output_router_logits : bool, optional
#             Whether to output router logits. The default is True.

#         Returns:
#         --------
#         x : torch.Tensor or Tuple

#             Output tensor of shape (batch_size, sequence_length, embed_dim). If `need_weights`, is ``True``,
#             output is a tuple of (x, attn). If `output_router_logits` is ``True``, the output will be a tuple
#             of (x, router_logits) or (x, attn, router_logts) depending on the value of `need_weights`.


#         .. _here:
#             https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
#         """
#         # pre-norm
#         residual = x
#         if self.pre_norm:
#             x = self.norm1(x)

#         # positional embeddings
#         x = self.embedding_dropout(self.positional_embeddings(x))

#         # attention
#         x = self.self_attn(
#             x,
#             x,
#             x,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#             attn_mask=attention_mask,
#         )
#         if need_weights:
#             x, attn = x
#         else:
#             x = x[0]
#         x = residual + self.dropout(x)

#         # post-norm
#         if not self.pre_norm:
#             x = self.norm1(x)

#         # pre-norm
#         residual = x
#         if self.pre_norm:
#             x = self.norm2(x)

#         # feedforward
#         x, router_tuple = self.mlp(x)
#         x = residual + self.dropout(x)

#         # post-norm
#         if not self.pre_norm:
#             x = self.norm2(residual + x)

#         # outputs
#         if need_weights:
#             return (x, attn, router_tuple)
#         return (x, router_tuple)


class HybridSparseTransformerLayer(nn.Module):
    """
    Hybrid sparse transformer layer, inspired by Snowflake's `Arctic model`_.
    The model has a dense transformer and a sparse residual connection.

    .. _Arctic model:
        https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        residual_ffn_dim: int,
        num_heads: int,
        num_experts: int,
        expert_capacity: int,
        max_length: int,
        num_shared_experts: int = 0,
        send_bos_to_all_experts: bool = True,
        top_k: int = 2,
        activation: str = "swiglu",
        expert_activation: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        expert_ffn_dropout: float = 0.0,
        token_embedding_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        router_dtype: str = "float32",
        router_bias: bool = False,
        router_jitter: float = 0.0,
        router_ignore_padding_tokens: bool = True,
        expert_choice_router: bool = False,
        pre_norm: bool = True,
        positional_embedding_type: str = "rotary",
    ):
        super().__init__()

        # positional embedding
        if positional_embedding_type.lower() == "rotary":
            self.positional_embeddings = RotaryPositionalEmbedding(
                embed_dim, max_length
            )
        elif positional_embedding_type.lower() == "relative":
            self.positional_embeddings = RelativePositionalEmbedding(
                embed_dim, max_length
            )
        else:
            raise ValueError(
                f"Invalid positional embedding type: {positional_embedding_type}. Valid options are 'rotary' or 'relative'."
            )
        self.embedding_dropout = nn.Dropout(token_embedding_dropout)

        # dense transformer
        self.dense_transformer = DenseTransformerLayer(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            max_length=max_length,
            dropout=dropout,
            attention_dropout=attention_dropout,
            token_embedding_dropout=token_embedding_dropout,
            layer_norm_eps=layer_norm_eps,
            activation=activation,
            positional_embedding_type=None,
            pre_norm=pre_norm,
        )

        # sparse residual connection
        self.residual_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.sparse_residual = SparseFFN(
            embed_dim=embed_dim,
            ffn_dim=residual_ffn_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            num_shared_experts=num_shared_experts,
            send_bos_to_all_experts=send_bos_to_all_experts,
            top_k=top_k,
            expert_ffn_dropout=expert_ffn_dropout,
            activation=expert_activation,
            router_dtype=router_dtype,
            router_bias=router_bias,
            router_jitter=router_jitter,
            router_ignore_padding_tokens=router_ignore_padding_tokens,
            router_class=TopKRouter,
            expert_class=Expert,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        output_router_logits: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Process the input hidden states.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        attn_mask : torch.Tensor, optional
            Attention mask of shape (batch_size * num_heads, sequence_length, sequence_length). The default is None.

        key_padding_mask : torch.Tensor, optional
            Mask of shape (batch_size, sequence_length). The default is None.

        need_weights : bool, optional
            Whether to return attention weights. The default is False.

            .. note::
                if `need_weights` is ``True``, the output will be a tuple of (x, attn). Also,
                nn.MultiHeadAttention will not be able to use the optimized torch implementation
                of ``scaled_dot_product_attention``. See `here`_ for more details.

        output_router_logits : bool, optional
            Whether to output router logits. The default is True.

        Returns:
        --------
        x : torch.Tensor or Tuple

            Output tensor of shape (batch_size, sequence_length, embed_dim). If `need_weights`, is ``True``,
            output is a tuple of (x, attn). If `output_router_logits` is ``True``, the output will be a tuple
            of (x, router_logits) or (x, attn, router_logts) depending on the value of `need_weights`.


        .. _here:
            https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        """
        # positional embeddings
        x = self.embedding_dropout(self.positional_embeddings(x))

        # residual connection
        residual, router_tuple = self.sparse_residual(self.residual_norm(x))

        # dense transformer
        x = self.dense_transformer(
            x,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        if need_weights:
            x, attn = x

        # add residual
        x = x + residual

        # outputs
        if need_weights:
            return (x, attn, router_tuple)
        return (x, router_tuple)


# class TransformerLayer(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         attention_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#         activation: str = "gelu",
#     ):
#         """
#         Transformer block with relative position embeddings and GELU activation.

#         Parameters
#         ----------
#         embed_dim : int
#             The input embedding dimension.

#         heads : int
#             The number of attention heads.

#         forward_expansion : int
#             The expansion factor for the feedforward network.

#         max_len : int
#             The maximum sequence length.

#         dropout : float
#             The dropout probability.
#         """
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

#         self.attention = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=attention_dropout,
#             batch_first=True,
#         )

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_dim, ffn_dim),
#             nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
#             nn.Linear(ffn_dim, embed_dim),
#         )

#         self.dropout = nn.Dropout(dropout)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#     ):
#         # pre-norm
#         residual = x
#         x = self.norm1(x)

#         # attention
#         x = self.attention(
#             x,
#             x,
#             x,
#             attn_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#         )
#         if need_weights:
#             x, weights = x
#         else:
#             x = x[0]
#         x = residual + self.dropout(x)

#         # pre-norm
#         residual = x
#         x = self.norm2(x)

#         # feedforward
#         x = self.feed_forward(x)
#         x = residual + self.dropout(x)

#         if need_weights:
#             return x, weights
#         return x


# class RoformerLayer(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_heads: int,
#         max_len: int,
#         dropout: float = 0.0,
#         attention_dropout: float = 0.0,
#         token_embedding_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#     ):
#         """
#         Transformer block with rotary embeddings and SwiGLU activation.

#         Parameters
#         ----------
#         embed_dim : int
#             The input embedding dimension.

#         heads : int
#             The number of attention heads.

#         forward_expansion : int
#             The expansion factor for the feedforward network.

#         max_len : int
#             The maximum sequence length.

#         dropout : float
#             The dropout probability.

#         attention_dropout : float
#             The dropout probability for the attention weights.

#         token_embedding_dropout : float
#             The dropout probability for the token embeddings.

#         layer_norm_eps : float
#             The epsilon value for the layer normalization.
#         """
#         super().__init__()
#         self.rotary_embedding = RotaryPositionalEmbedding(embed_dim, max_len)

#         self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
#         self.attention = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=attention_dropout,
#             batch_first=True,
#         )

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_dim, ffn_dim),
#             SwiGLU(),
#             nn.Linear(ffn_dim // 2, embed_dim),  # adjusted for SwiGLU
#         )

#         self.dropout = nn.Dropout(dropout)
#         self.token_embedding_dropout = nn.Dropout(token_embedding_dropout)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#     ):
#         # pre-norm
#         residual = x
#         x = self.token_embedding_dropout(self.norm1(x))

#         # rotary embeddings
#         x = self.rotary_embedding(x)

#         # attention
#         x = self.attention(
#             x,
#             x,
#             x,
#             attn_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#         )
#         if need_weights:
#             x, weights = x
#         else:
#             x = x[0]
#         x = residual + self.dropout(x)

#         # pre-norm
#         residual = x
#         x = self.norm2(x)

#         # feedforward
#         x = self.feed_forward(x)
#         x = residual + self.dropout(x)

#         if need_weights:
#             return x, weights
#         return x


# class SparseMLP(nn.Module):
#     """
#     Implementation of the Switch Transformers Sparse MLP module.

#     Parameters:
#     -----------
#     config : BalmMoEConfig
#         Model configuration class with all the parameters of the model.
#         Initializing with a config file does not load the weights associated with the model, only the
#         configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

#     router_class : nn.Module, optional
#         Router class to use. The default is ``TopKRouter``.

#     expert_class : nn.Module, optional
#         Expert class to use. The default is ``Expert``.

#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_experts: int,
#         expert_capacity: int,
#         num_shared_experts: int = 0,
#         top_k: int = 1,
#         expert_activation: str = "gelu",
#         expert_ffn_dropout: float = 0.0,
#         router_dtype: str = "float32",
#         router_bias: bool = False,
#         router_jitter: float = 0.0,
#         router_ignore_padding_tokens: bool = True,
#         router_class: nn.Module = TopKRouter,
#         expert_class: nn.Module = Expert,
#     ):
#         super().__init__()
#         self.router = router_class(
#             embed_dim=embed_dim,
#             num_experts=num_experts,
#             expert_capacity=expert_capacity,
#             top_k=top_k,
#             num_shared_experts=num_shared_experts,
#             dtype=router_dtype,
#             bias=router_bias,
#             jitter=router_jitter,
#             ignore_padding_tokens=router_ignore_padding_tokens,
#         )
#         self.experts = nn.ModuleDict()
#         for idx in range(num_experts):
#             self.experts[f"expert_{idx}"] = expert_class(
#                 embed_dim=embed_dim,
#                 ffn_dim=ffn_dim,
#                 dropout_rate=expert_ffn_dropout,
#                 activation=expert_activation,
#             )

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
#         """
#         Route tokens to experts and process them.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         x : torch.Tensor
#             Output tensor of shape (batch_size, sequence_length, embed_dim).
#         """
#         # get the router mask, probabilities, and logits
#         expert_mask, router_probs, router_logits = self.router(x)
#         expert_outputs = []

#         for idx, expert in self.experts.items():
#             int_idx = int(idx.split("_")[-1])
#             token_indices = expert_mask[..., int_idx].bool()
#             expert_output = expert(x[token_indices]).to(x.dtype)
#             expanded_output = torch.zeros_like(x)
#             expanded_output[token_indices] = expert_output
#             expert_outputs.append(expanded_output)

#         # Combine the outputs from the selected tokens for each expert
#         x = torch.stack(expert_outputs, dim=-1) * expert_mask.unsqueeze(-2)
#         x = x.sum(dim=-1)

#         return x, (router_logits, expert_mask)


# class SparseTransformerLayer(nn.Module):
#     """
#     BALM transformer layer with Mixture of Experts. Approximately follows the ESM-2
#     implementation, but differs in a few ways:
#         - includes (optional) dropout for self-attention and feedforward layers
#         - normalize **after**, not before, the self-attention and feedforward layers
#         - we don't use rotary embeddings, which aren't (yet?) compatible with
#           torch's optimized implementation of ``nn.MultiheadAttention``

#     Parameters:
#     -----------
#     config : BalmMoEConfig
#         Model configuration class with all the parameters of the model.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_heads: int,
#         num_experts: int,
#         expert_capacity: int,
#         num_shared_experts: int = 0,
#         top_k: int = 1,
#         expert_activation: str = "gelu",
#         dropout: float = 0.1,
#         attention_dropout: float = 0.0,
#         expert_ffn_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#         router_dtype: str = "float32",
#         router_bias: bool = False,
#         router_jitter: float = 0.0,
#         router_ignore_padding_tokens: bool = True,
#         router_class: nn.Module = TopKRouter,
#         expert_class: nn.Module = Expert,
#         # config: BalmMoEConfig,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.ffn_dim = ffn_dim
#         self.num_heads = num_heads
#         # self.attention_dropout = attention_dropout
#         # self.expert_ffn_dropout = expert_ffn_dropout
#         self.layer_norm_eps = layer_norm_eps

#         # can't use rotary embeddings with nn.MultiheadAttention
#         # see: https://discuss.pytorch.org/t/is-there-a-way-to-implement-rope-around-nn-multiheadattention-somehow/175051
#         # it is possible to use rotary embeddings with F.scaled_dot_product_attention,
#         # but it's not clear that it's worth the effort
#         # see: https://github.com/pytorch/pytorch/issues/97899 for an example
#         # self.use_rotary_embeddings = use_rotary_embeddings

#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=self.num_heads,
#             dropout=attention_dropout,
#             batch_first=True,
#         )

#         self.mlp = SparseMLP(
#             embed_dim=self.embed_dim,
#             ffn_dim=self.ffn_dim,
#             num_experts=num_experts,
#             num_shared_experts=num_shared_experts,
#             top_k=top_k,
#             expert_capacity=expert_capacity,
#             expert_activation=expert_activation,
#             expert_ffn_dropout=expert_ffn_dropout,
#             router_dtype=router_dtype,
#             router_bias=router_bias,
#             router_jitter=router_jitter,
#             router_ignore_padding_tokens=router_ignore_padding_tokens,
#             router_class=router_class,
#             expert_class=expert_class,
#         )
#         self.dropout = nn.Dropout(dropout)

#         self.norm1 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
#         self.norm2 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#         output_router_logits: bool = True,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         attn_mask : torch.Tensor, optional
#             Attention mask of shape (batch_size * num_heads, sequence_length, sequence_length). The default is None.

#         key_padding_mask : torch.Tensor, optional
#             Mask of shape (batch_size, sequence_length). The default is None.

#         need_weights : bool, optional
#             Whether to return attention weights. The default is False.

#             .. note::
#                 if `need_weights` is ``True``, the output will be a tuple of (x, attn). Also,
#                 nn.MultiHeadAttention will not be able to use the optimized torch implementation
#                 of ``scaled_dot_product_attention``. See `here`_ for more details.

#         output_router_logits : bool, optional
#             Whether to output router logits. The default is True.

#         Returns:
#         --------
#         x : torch.Tensor or Tuple

#             Output tensor of shape (batch_size, sequence_length, embed_dim). If `need_weights`, is ``True``,
#             output is a tuple of (x, attn). If `output_router_logits` is ``True``, the output will be a tuple
#             of (x, router_logits) or (x, attn, router_logts) depending on the value of `need_weights`.


#         .. _here:
#             https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
#         """
#         # attention
#         residual = x
#         x = self.self_attn(
#             query=x,
#             key=x,
#             value=x,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#             attn_mask=attention_mask,
#         )
#         if need_weights:
#             x, attn = x
#         else:
#             x = x[0]
#         x = residual + self.dropout(x)
#         x = self.norm1(x)

#         # sparse feedforward
#         residual = x
#         x, router_tuple = self.mlp(x)  # router_tuple is (router_logits, expert_index)
#         x = self.dropout(x)
#         x = self.norm2(residual + x)
#         if output_router_logits and router_tuple is not None:
#             if need_weights:
#                 return (x, attn, router_tuple)
#             return (x, router_tuple)
#         if need_weights:
#             return (x, attn)
#         return x


# class HybridSparseTransformerLayer(nn.Module):
#     """
#     Hybrid sparse transformer layer. Inspired by Snowflake's `Arctic model`_.

#     .. _Arctic model:
#         https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         residual_ffn_dim: int,
#         num_heads: int,
#         max_length: int,
#         num_experts: int,
#         expert_capacity: int,
#         num_shared_experts: int = 0,
#         top_k: int = 2,
#         activation: str = "swiglu",
#         expert_activation: str = "swiglu",
#         dropout: float = 0.1,
#         attention_dropout: float = 0.0,
#         expert_ffn_dropout: float = 0.0,
#         token_embedding_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#         positional_embedding_type: str = "rotary",
#         pre_norm: bool = True,
#         router_dtype: str = "float32",
#         router_bias: bool = False,
#         router_jitter: float = 0.0,
#         router_ignore_padding_tokens: bool = True,
#         expert_choice_router: bool = True,
#         # config: BalmMoEConfig,
#     ):
#         super().__init__()
#         self.dense_transformer = DenseTransformerLayer(
#             embed_dim,
#             ffn_dim,
#             num_heads,
#             max_length,
#             dropout=dropout,
#             attention_dropout=attention_dropout,
#             token_embedding_dropout=token_embedding_dropout,
#             layer_norm_eps=layer_norm_eps,
#             activation=activation,
#             positional_embedding_type=positional_embedding_type,
#             pre_norm=pre_norm,
#         )
#         self.sparse_residual = SparseMLP(
#             embed_dim=embed_dim,
#             ffn_dim=ffn_dim,
#             num_experts=num_experts,
#             num_shared_experts=num_shared_experts,
#             top_k=top_k,
#             expert_capacity=expert_capacity,
#             activation=expert_activation,
#             expert_ffn_dropout=expert_ffn_dropout,
#             router_dtype=router_dtype,
#             router_bias=router_bias,
#             router_jitter=router_jitter,
#             router_ignore_padding_tokens=router_ignore_padding_tokens,
#             router_class=ExpertChoiceRouter if expert_choice_router else TopKRouter,
#             expert_class=Expert,
#         )
#         self.residual_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#         output_router_logits: bool = True,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         attn_mask : torch.Tensor, optional
#             Attention mask of shape (batch_size * num_heads, sequence_length, sequence_length). The default is None.

#         key_padding_mask : torch.Tensor, optional
#             Mask of shape (batch_size, sequence_length). The default is None.

#         need_weights : bool, optional
#             Whether to return attention weights. The default is False.

#             .. note::
#                 if `need_weights` is ``True``, the output will be a tuple of (x, attn). Also,
#                 nn.MultiHeadAttention will not be able to use the optimized torch implementation
#                 of ``scaled_dot_product_attention``. See `here`_ for more details.

#         output_router_logits : bool, optional
#             Whether to output router logits. The default is True.

#         Returns:
#         --------
#         x : torch.Tensor or Tuple

#             Output tensor of shape (batch_size, sequence_length, embed_dim). If `need_weights`, is ``True``,
#             output is a tuple of (x, attn). If `output_router_logits` is ``True``, the output will be a tuple
#             of (x, router_logits) or (x, attn, router_logts) depending on the value of `need_weights`.


#         .. _here:
#             https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
#         """
#         # sparse residual connection
#         residual, (router_logits, expert_mask) = self.sparse_residual(
#             self.residual_norm(x)
#         )

#         # dense transformer
#         x = self.dense_transformer(
#             x,
#             attention_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#         )
#         if need_weights:
#             x, attn = x
#         else:
#             x = x[0]
#         x = x + residual
#         if need_weights:
#             if output_router_logits:
#                 return (x, attn, router_logits)
#             return (x, attn)
#         if output_router_logits:
#             return (x, router_logits)
#         return x


# class SparseRoformerLayer(nn.Module):
#     """
#     Sparse Roformer layer.


#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         num_heads: int,
#         num_experts: int,
#         max_len: int,
#         expert_capacity: int,
#         num_shared_experts: int = 0,
#         top_k: int = 1,
#         expert_activation: str = "gelu",
#         dropout: float = 0.1,
#         attention_dropout: float = 0.0,
#         expert_ffn_dropout: float = 0.0,
#         token_embedding_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#         router_dtype: str = "float32",
#         router_bias: bool = False,
#         router_jitter: float = 0.0,
#         router_ignore_padding_tokens: bool = True,
#         router_class: nn.Module = TopKRouter,
#         expert_class: nn.Module = Expert,
#     ):
#         super().__init__()
#         self.rotary_embedding = RotaryPositionalEmbedding(embed_dim, max_len)

#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.attention = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=attention_dropout,
#             batch_first=True,
#         )

#         self.mlp = SparseMLP(
#             embed_dim=embed_dim,
#             ffn_dim=ffn_dim,
#             num_experts=num_experts,
#             num_shared_experts=num_shared_experts,
#             expert_capacity=expert_capacity,
#             top_k=top_k,
#             expert_activation=expert_activation,
#             expert_ffn_dropout=expert_ffn_dropout,
#             router_dtype=router_dtype,
#             router_bias=router_bias,
#             router_jitter=router_jitter,
#             router_ignore_padding_tokens=router_ignore_padding_tokens,
#             router_class=router_class,
#             expert_class=expert_class,
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.token_embedding_dropout = nn.Dropout(token_embedding_dropout)

#         self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

#     def forward(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False,
#         output_router_logits: bool = True,
#     ):
#         # pre-norm
#         query_norm = self.token_embedding_dropout(self.norm1(query))
#         key_norm = self.token_embedding_dropout(self.norm1(key))
#         value_norm = self.token_embedding_dropout(self.norm1(value))

#         # rotary embeddings
#         query_rot = self.rotary_embedding(query_norm)
#         key_rot = self.rotary_embedding(key_norm)
#         value_rot = self.rotary_embedding(value_norm)

#         # attention
#         x = self.attention(
#             query_rot,
#             key_rot,
#             value_rot,
#             attn_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#         )
#         # x = query + self.dropout(x)
#         if need_weights:
#             x, attn = x
#         else:
#             x = x[0]
#         x = query + self.dropout(x)

#         # pre-norm
#         residual = x
#         x = self.norm2(x)

#         # sparse feedforward
#         x, router_tuple = self.mlp(x)
#         x = residual + self.dropout(x)

#         if output_router_logits and router_tuple is not None:
#             if need_weights:
#                 return (x, attn, router_tuple)
#             return (x, router_tuple)
#         if need_weights:
#             return (x, attn)
#         return x
