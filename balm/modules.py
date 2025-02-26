# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_activation_fn
from .embedding import RotaryPositionalEmbedding
from .router import ExpertChoiceRouter, TopKRouter

__all__ = [
    # layers
    "DenseTransformerLayer",
    "SparseTransformerLayer",
    # "HybridSparseTransformerLayer",
    "SparseFFN",
    "DenseFFN",
    "SwigluFFN",
    "Expert",
    # heads
    "BalmLMHead",
    "BalmSequenceClassificationHead",
    "BalmTokenClassificationHead",
]


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

    activation: str, default="gelu"
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
        activation: str = "gelu",
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

    Parameters:
    -----------
    model_dim: int
        Token embedding dimension.

    ffn_dim: int, default=None
        Feed-forward network dimension. If not provided, it will be set to 4x the model dimension.

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
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_linear = nn.Linear(model_dim, ffn_dim, bias=bias)
        self.value_linear = nn.Linear(model_dim, ffn_dim, bias=bias)
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwigluFFN layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim).

        """
        gate = self.gate_linear(x)
        value = self.value_linear(x)
        x = value * F.silu(gate)
        return self.dropout(self.wo(x))


class SparseFFN(nn.Module):
    """
    Sparse Mixture of Experts layer with capacity constraints.

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

    router_type: str, default="topk"
        Router type. Options are "topk" or "expert choice".

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
        router_type: str = "topk",
        activation: str = "swiglu",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # router
        self.router_type = router_type
        if router_type == "topk":
            self.router = TopKRouter(model_dim, num_experts)
        elif router_type == "expert choice":
            self.router = ExpertChoiceRouter(model_dim, num_experts)
        else:
            raise ValueError(f"Invalid router type: {router_type}")

        # experts
        if activation.lower() == "swiglu":
            expert = partial(
                SwigluFFN,
                model_dim=model_dim,
                ffn_dim=ffn_dim,
                bias=bias,
                dropout=dropout,
            )
        else:
            expert = partial(
                DenseFFN,
                model_dim=model_dim,
                ffn_dim=ffn_dim,
                bias=bias,
                dropout=dropout,
                activation=activation,
            )
        self.experts = nn.ModuleList([expert() for _ in range(num_experts)])
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
        Determine expert capacity.

        Parameters:
        -----------
        num_tokens : int
            Number of tokens in the batch.

        Returns:
        --------
        capacity : int
            Expert capacity.
        """
        if self.capacity_multiplier:
            return int(self.capacity_multiplier * num_tokens)
        return self.absolute_capacity

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the SparseFFN layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
        --------
        output : torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim).

        router_tuple : Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the following:
            - router_logits : torch.Tensor
                Router logits of shape (batch_size, seq_len, num_experts).
            - router_indices : torch.Tensor
                Router indices of shape (batch_size, seq_len, num_experts).
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.view(-1, d_model)  # ==> (num_tokens, d_model)

        capacity = self._compute_capacity(num_tokens)
        # logits are shape (num_tokens, num_experts)
        # probs and indices are shape (num_experts, expert_capacity)
        router_logits, router_probs, expert_probs, expert_indices = self.router(
            x_flat, k=self.k, expert_capacity=capacity
        )

        output = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            # get token indices and probs for current expert
            token_indices = expert_indices[expert_idx]  # ==> (expert_capacity,)
            token_probs = expert_probs[expert_idx]  # ==> (expert_capacity,)
            # remove padding (in undersubscribed experts, empty slots are filled with -1)
            valid_token_mask = token_indices >= 0
            valid_token_indices = token_indices[valid_token_mask]
            valid_token_probs = token_probs[valid_token_mask]
            if valid_token_indices.numel() == 0:  # no valid tokens for this expert
                continue
            # get expert input and weights
            expert_input = x_flat[valid_token_indices]
            weights = valid_token_probs
            # compute expert output
            expert_output = expert(expert_input) * weights.unsqueeze(1)
            # accumulate expert output
            output[valid_token_indices] += expert_output

        output = output.view(batch_size, seq_len, d_model)
        return output, (router_logits, router_probs, expert_probs, expert_indices)


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
        Position embedding type. Only used if rotary embeddings are specified
        (i.e. `position_embedding_type="rotary"`).
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
        # use attention's combined projection weights/biases
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
    """
    Sparse Transformer layer.

    Parameters
    ----------
    model_dim: int
        Token embedding dimension.

    ffn_dim: int
        Feed-forward dimension.

    num_heads: int
        Number of attention heads.

    num_experts: int
        Number of experts in SparseFFN.

    max_capacity: Union[int, float]
        Expert capacity (int = absolute, float = multiplier).

    k: int, default=1
        Number of experts per token.

    router_type: str, default="topk"
        Router type. Options are "topk" or "expert choice".

    activation: str, default="swiglu"
        Activation function.

    bias: bool, default=True
        Whether to use bias.

    dropout: float, default=0.1
        Dropout probability.

    position_embedding_type: str, default="rotary"
        Position embedding type. Only used if rotary embeddings are specified
        (i.e. `position_embedding_type="rotary"`).

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
        router_type: str = "topk",
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
            router_type=router_type,
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
        Forward pass for the SparseTransformerLayer.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)

        padding_mask: Optional[torch.Tensor], default=None
            Boolean mask indicating padded positions (batch_size, seq_len)

        need_weights: bool, default=False
            Whether to return attention values.

        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
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


# class HybridSparseTransformerLayer(nn.Module):
#     """
#     Hybrid sparse transformer layer, inspired by Snowflake's `Arctic model`_.
#     The model has a dense transformer and a sparse residual connection.

#     .. _Arctic model:
#         https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         ffn_dim: int,
#         residual_ffn_dim: int,
#         num_heads: int,
#         num_experts: int,
#         expert_capacity: int,
#         max_length: int,
#         num_shared_experts: int = 0,
#         send_bos_to_all_experts: bool = True,
#         top_k: int = 2,
#         activation: str = "swiglu",
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
#         expert_choice_router: bool = False,
#         pre_norm: bool = True,
#         positional_embedding_type: str = "rotary",
#     ):
#         super().__init__()

#         # positional embedding
#         if positional_embedding_type.lower() == "rotary":
#             self.positional_embeddings = RotaryPositionalEmbedding(
#                 embed_dim, max_length
#             )
#         elif positional_embedding_type.lower() == "relative":
#             self.positional_embeddings = RelativePositionalEmbedding(
#                 embed_dim, max_length
#             )
#         else:
#             raise ValueError(
#                 f"Invalid positional embedding type: {positional_embedding_type}. Valid options are 'rotary' or 'relative'."
#             )
#         self.embedding_dropout = nn.Dropout(token_embedding_dropout)

#         # dense transformer
#         self.dense_transformer = DenseTransformerLayer(
#             embed_dim=embed_dim,
#             ffn_dim=ffn_dim,
#             num_heads=num_heads,
#             max_length=max_length,
#             dropout=dropout,
#             attention_dropout=attention_dropout,
#             token_embedding_dropout=token_embedding_dropout,
#             layer_norm_eps=layer_norm_eps,
#             activation=activation,
#             positional_embedding_type=None,
#             pre_norm=pre_norm,
#         )

#         # sparse residual connection
#         self.residual_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
#         self.sparse_residual = SparseFFN(
#             embed_dim=embed_dim,
#             ffn_dim=residual_ffn_dim,
#             num_experts=num_experts,
#             expert_capacity=expert_capacity,
#             num_shared_experts=num_shared_experts,
#             send_bos_to_all_experts=send_bos_to_all_experts,
#             top_k=top_k,
#             expert_ffn_dropout=expert_ffn_dropout,
#             activation=expert_activation,
#             router_dtype=router_dtype,
#             router_bias=router_bias,
#             router_jitter=router_jitter,
#             router_ignore_padding_tokens=router_ignore_padding_tokens,
#             router_class=TopKRouter,
#             expert_class=Expert,
#         )

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
#         # positional embeddings
#         x = self.embedding_dropout(self.positional_embeddings(x))

#         # residual connection
#         residual, router_tuple = self.sparse_residual(self.residual_norm(x))

#         # dense transformer
#         x = self.dense_transformer(
#             x,
#             attention_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#         )
#         if need_weights:
#             x, attn = x

#         # add residual
#         x = x + residual

#         # outputs
#         if need_weights:
#             return (x, attn, router_tuple)
#         return (x, router_tuple)


# cla
