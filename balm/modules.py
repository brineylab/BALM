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
from transformers import PretrainedConfig

__all__ = [
    # layers
    "DenseTransformerLayer",
    "SparseTransformerLayer",
    "SparseFFN",
    "DenseFFN",
    "GluFFN",
    # heads
    "BalmLMHead",
    "BalmSequenceClassificationHead",
    # "BalmTokenClassificationHead",
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

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation_fn(config.mlm_activation)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        x = self.decoder(x) + self.bias # proj to vocab size
        return x


class BalmSequenceClassificationHead(nn.Module):
    """
    Head for sequence-level classification tasks.

    Parameters
    ----------
    config: PretrainedConfig
        Model config.

    """

    def __init__(self, config: PretrainedConfig
    ):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # activation
        self.activation = get_activation_fn(config.classifier_activation)

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


# class BalmTokenClassificationHead(nn.Module):
#     """
#     Head for token-level classification tasks.

#     Parameters
#         ----------
#         config: PretrainedConfig
#             Model config.
#     """

#     def __init__(self, config: PretrainedConfig):
#         super().__init__()
#         self.dropout = nn.Dropout(config.classifier_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
#         """
#         BalmTokenClassificationHead forward pass.

#         Parameters
#         ----------
#         features : torch.Tensor
#             Features tensor of shape (batch_size, sequence_length, hidden_size).

#         Returns
#         -------
#         x : torch.Tensor
#             Output tensor of shape (batch_size, sequence_length, num_labels).

#         """
#         x = self.dropout(features)
#         x = self.out_proj(x)
#         return x


# =================================
#
#             FFNs
#
# =================================


class DenseFFN(nn.Module):
    """
    Standard (dense) feed-forward network, used for RELU and GELU activations.

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

    Input shape: (batch_size, seq_len, model_dim)
    Output shape: (batch_size, seq_len, model_dim)

    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int = None,
        bias: bool = True,
        activation: str,
    ):
        super().__init__()
        self.wi = nn.Linear(model_dim, ffn_dim, bias=bias) # intermediate dense
        self.activation = get_activation_fn(activation)
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias) # output dense

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
        x = self.wo(x)
        return x


class GluFFN(nn.Module):
    """
    GLU (dense) feed-forward network, used for GLU, SwiGLU, GeGLU, and ReGLU activations.
    Replaces the first linear layer and activation with GLU, as described `here`_.

    .. _here:
        https://arxiv.org/pdf/2002.05202

    Parameters:
    -----------
    model_dim: int
        Token embedding dimension.

    ffn_dim: int, default=None
        Feed-forward network dimension. If not provided, it will be set to 4x the model dimension.

    bias: bool, default=True
        Whether to use bias.

    Input shape: (batch_size, seq_len, model_dim)
    Output shape: (batch_size, seq_len, model_dim)
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int = None,
        bias: bool = True,
        activation: str,
    ):
        super().__init__()
        self.activation = get_activation_fn(
            activation, 
            input_dim=model_dim, 
            output_dim=ffn_dim,
            bias=bias
        )
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias) # output dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GluFFN layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim).

        """
        # activation handles model_dim -> ffn_dim
        x = self.activation(x)
        x = self.wo(x)
        return x


class SparseFFN(nn.Module):
    """
    Sparse Mixture of Experts layer with capacity constraints.

    Parameters:
    -----------
    model_dim: int
        Token embedding dimension.

    ffn_dim: int, default=None
        Feed-forward network dimension. If not provided, it will be set to 4x the model dimension.

    expert_bias: bool, default=True
        Whether to use bias in the expert FFN.

    num_experts: int
        Number of experts.
    
    num_shared_experts: int
        Number of shared experts (which receive all the tokens).
    
    expert_capacity_type : str
        The type of expert capacity to use. 
        If "absolute": tokens per expert; if "multiplier": capacity = multiplier * max_position_embeddings

    expert_capacity: Union[int, float]
        Expert capacity, either absolute or multiplier based on expert_capacity_type

    k: int, default=1
        Number of experts per token. Used in "topk" routing only. 

    router_type: str, default="topk"
        Router type. Options are "topk" or "expert choice".
    
    router_bias: bool, default=False
        Whether to use bias in the router.
    
    router_dtype: torch.dtype
        Data type to use for softmax of the router.

    router_jitter: float
        Jitter to apply to inputs of the router.

    expert_activation: str, default="swiglu"
        Activation function to use.
    
    expert_bias: bool, default=True
        Whether to use bias in the experts.

    Input shape: (batch_size, seq_len, model_dim)
    Output shape: (batch_size, seq_len, model_dim)
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_experts: int,
        num_shared_experts: int,
        expert_capacity_type: str,
        expert_capacity: Union[int, float],
        k: int = 1,
        router_type: str = "topk",
        router_bias: bool = False,
        router_dtype: str = "float32",
        router_jitter: float = 0.0,
        expert_activation: str = "swiglu",
        expert_bias: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts - num_shared_experts # subtract shared experts
        self.num_shared_experts = num_shared_experts
        self.k = k
        
        # router
        self.router_type = router_type
        self.router_jitter = router_jitter
        if router_type == "topk":
            self.router = TopKRouter(
                d_model=model_dim, 
                num_experts=self.num_experts,
                router_bias=router_bias,
                router_dtype=router_dtype,
            )
        elif router_type == "expert choice":
            self.router = ExpertChoiceRouter(
                d_model=model_dim, 
                num_experts=self.num_experts,
                router_bias=router_bias,
                router_dtype=router_dtype,
            )
        else:
            raise ValueError(f"Invalid router type: {router_type}")

        # experts
        ffn_class = GluFFN if "glu" in expert_activation else DenseFFN
        expert = partial(
            ffn_class,
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            bias=expert_bias,
            activation=expert_activation,
        )
        self.experts = nn.ModuleList([expert() for _ in range(self.num_experts)]) # excluding shared expert(s)
        self.shared_experts = nn.ModuleList([expert() for _ in range(num_shared_experts)])

        # expert capacity (applied to non-shared experts)
        if expert_capacity < 0:
            self.expert_capacity = -1
        elif expert_capacity_type == "absolute":
            self.expert_capacity = expert_capacity
        else: # need num_tokens to compute
            self.capacity_multiplier = expert_capacity
            self.expert_capacity = None

    def _compute_multiplier_capacity(self, num_tokens: int) -> int:
        """
        Determine expert capacity, if capacity multiplier was provided.

        Parameters:
        -----------
        num_tokens : int
            Number of tokens in the batch.

        Returns:
        --------
        capacity : int
            Expert capacity.
        """
        return int(self.capacity_multiplier * num_tokens / self.num_experts)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the SparseFFN layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim).

        padding_mask: Optional[torch.Tensor], default=None
            Boolean mask indicating padded positions (batch_size, seq_len)

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

        # add jitter if training, before reshaping logits
        if self.training and self.router_jitter > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.router_jitter, 1.0 + self.router_jitter)

        # flatten logits & padding mask
        x_flat = x.view(-1, d_model)  # ==> (num_tokens, d_model)
        padding_flat = padding_mask.view(-1) if padding_mask is not None else None

        # expert capacity
        capacity = self._compute_multiplier_capacity(num_tokens) if self.expert_capacity is None else self.expert_capacity

        # logits are shape (num_tokens, num_experts)
        # probs and indices are shape (num_experts, expert_capacity)
        router_logits, router_probs, expert_probs, expert_indices = self.router(
            x_flat, 
            padding_mask=padding_flat,
            k=self.k, 
            expert_capacity=capacity,
        )

        # clone hidden states
        # this passes hidden states unchanged for tokens that aren't sent to any expert
        output = x_flat.clone()

        # experts (excluding shared expert)
        for expert_idx, expert in enumerate(self.experts):
            # get token indices and probs for current expert ==> (expert_capacity,)
            token_indices = expert_indices[expert_idx]
            token_probs = expert_probs[expert_idx]
            
            # remove tokens that were not selected
            # (in undersubscribed experts, empty slots are filled with -1)
            valid_token_mask = token_indices >= 0
            valid_token_indices = token_indices[valid_token_mask]
            valid_token_probs = token_probs[valid_token_mask]

            # no valid tokens for this expert
            if valid_token_indices.numel() == 0:
                continue
            
            # get expert input
            expert_input = x_flat[valid_token_indices]
            
            # compute expert output
            # scale output by routing probability
            expert_output = expert(expert_input) * valid_token_probs.unsqueeze(1)
            
            # accumulate expert output
            output[valid_token_indices] += expert_output
            
        # shared expert(s)
        for expert_idx, expert in enumerate(self.shared_experts):
            # get expert input for all tokens
            expert_input = x_flat
            
            # compute expert output
            # don't scale by routing probability because not passed through router
            expert_output = expert(expert_input)
            
            # accumulate expert output
            output += expert_output

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

        if model_dim % num_heads != 0:
            raise ValueError(f"Model dim ({model_dim}) must be divisible by num_heads ({num_heads}).")
        
        self.head_dim =  model_dim // num_heads
        self.num_heads = num_heads
        self.all_head_size = self.num_heads * self.head_dim

        # embeddings
        self.rotary_embed = (
            RotaryPositionalEmbedding(self.head_dim) if position_embedding_type == "rotary" else None
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
    config: PretrainedConfig
        Model config.

    Input shape: (batch_size, seq_len, model_dim)
    Output shape: (batch_size, seq_len, model_dim)

    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # attention
        self.attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = SelfAttention(
            model_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type=config.position_embedding_type,
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # FFN
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the DenseTransformerLayer.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, model_dim)

        padding_mask: Optional[torch.Tensor], default=None
            Boolean mask indicating padded positions (batch_size, seq_len)

        need_weights: bool, default=False
            Whether to return attention values.
            
            .. warning::
                If ``need_weights`` is ``True``, torch can't use optimized SDPA.
                See https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (batch_size, seq_len, model_dim)
        """

        # invert padding mask and convert to boolean, since the 🤗 DataCollatorForLanguageModeling
        # uses 0 for padding tokens and 1 for other tokens, but we want True for padding tokens and
        # False for other tokens
        if padding_mask is not None:
            padding_mask = 1 - padding_mask
            padding_mask = padding_mask.bool()

        # attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_out = self.attention(
            x,
            padding_mask=padding_mask,
            need_weights=need_weights,
        )
        attn_out, attn_vals = attn_out if need_weights else (attn_out[0], None)
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

    Input shape: (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # attention
        self.attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = SelfAttention(
            model_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            position_embedding_type=config.position_embedding_type,
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # sparse FFN
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.sparse_ffn = SparseFFN(
            model_dim=config.hidden_size,
            ffn_dim=config.intermediate_size,
            expert_activation=config.expert_activation,
            expert_bias=config.expert_bias,
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            expert_capacity_type=config.expert_capacity_type,
            expert_capacity=config.expert_capacity,
            k=config.num_experts_per_tok,
            router_type=config.router_type,
            router_bias=config.router_bias,
            router_dtype=config.router_dtype,
            router_jitter=config.router_jitter,
        )
        self.ffn_dropout = nn.Dropout(config.expert_dropout)

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
            
            .. warning::
                If ``need_weights`` is ``True``, torch can't use optimized SDPA.
                See https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

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

        # attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_out = self.attention(
            x,
            padding_mask=padding_mask,
            need_weights=need_weights,
        )
        attn_out, attn_vals = attn_out if need_weights else (attn_out[0], None)
        x = residual + self.attn_dropout(attn_out)

        # sparse FFN
        residual = x
        x = self.ffn_layer_norm(x)
        ffn_out, router_tuple = self.sparse_ffn(
            x,
            padding_mask=padding_mask
        )
        x = residual + self.ffn_dropout(ffn_out)

        return (x, attn_vals, router_tuple) if need_weights else (x, router_tuple)
