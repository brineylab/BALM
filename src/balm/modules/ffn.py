# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Tuple, Union, List

import torch
import torch.nn as nn

from .activation import get_activation_fn
from .router import ExpertChoiceRouter, TopKRouter, TopPRouter

__all__ = ["SparseFFN", "DenseFFN", "GluFFN"]

ROUTER_MAPPING = {
    "top-k": TopKRouter,
    "top-p": TopPRouter,
    "expert-choice": ExpertChoiceRouter,
}


class DenseFFN(nn.Module):
    """
    Standard (dense) feed-forward network, used for RELU and GELU activations.

    Parameters:
    -----------
    model_dim: int
        Token embedding dimension.
    ffn_dim: int
        Feed-forward network dimension. If not provided, it will be set to 4x the model dimension.
    bias: bool
        Whether to use bias.
    activation: str
        Activation function to use.
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        bias: bool,
        activation: str,
    ):
        super().__init__()
        self.wi = nn.Linear(model_dim, ffn_dim, bias=bias)  # intermediate dense
        self.activation = get_activation_fn(activation)
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias)  # output dense

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
        Feed-forward network dimension.
    bias: bool, default=True
        Whether to use bias.
    activation: str
        Activation function to use.
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        bias: bool,
        activation: str,
    ):
        super().__init__()
        self.activation = get_activation_fn(
            activation, input_dim=model_dim, output_dim=ffn_dim, bias=bias
        )
        self.wo = nn.Linear(ffn_dim, model_dim, bias=bias)  # output dense

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
    expert_ffn_dims: List
        List of the feed-forward dimensions for each expert.
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
        Number of experts per token. Used in "top-k" routing only.
    router_type: str, default="top-k"
        Router type. Options are "top-k", "top-p", or "expert-choice".
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
    """

    def __init__(
        self,
        model_dim: int,
        expert_ffn_dims: List,
        shared_ffn_dim: int,
        num_experts: int,
        num_shared_experts: int,
        expert_capacity_type: str,
        expert_capacity: Union[int, float],
        k: int,
        top_p_threshold: float,
        router_type: str = "top-k",
        router_bias: bool = False,
        router_dtype: str = "float32",
        router_jitter: float = 0.0,
        expert_activation: str = "swiglu",
        expert_bias: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts - num_shared_experts  # subtract shared experts
        self.num_shared_experts = num_shared_experts
        self.router_jitter = router_jitter

        # router
        router_class = ROUTER_MAPPING[router_type]
        self.router = router_class(
            d_model=model_dim,
            num_experts=self.num_experts,
            router_bias=router_bias,
            router_dtype=router_dtype,
            k=k,
            top_p_threshold=top_p_threshold,
        )

        # experts
        ffn_class = GluFFN if "glu" in expert_activation else DenseFFN
        self.experts = nn.ModuleList(
            [
                ffn_class(
                    model_dim=model_dim,
                    ffn_dim=ffn_dim,
                    bias=expert_bias,
                    activation=expert_activation,
                )
                for ffn_dim in expert_ffn_dims
            ]
        )  # excluding shared expert(s)
        self.shared_experts = nn.ModuleList(
            [
                ffn_class(
                    model_dim=model_dim,
                    ffn_dim=shared_ffn_dim,
                    bias=expert_bias,
                    activation=expert_activation,
                )
                for _ in range(self.num_shared_experts)
            ]
        )

        # expert capacity (applied to non-shared experts)
        if expert_capacity < 0:
            self.expert_capacity = -1
        elif expert_capacity_type == "absolute":
            self.expert_capacity = expert_capacity
        else:  # need num_tokens to compute
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
            x *= torch.empty_like(x).uniform_(
                1.0 - self.router_jitter, 1.0 + self.router_jitter
            )

        # flatten logits
        x_flat = x.view(-1, d_model)  # ==> (num_tokens, d_model)

        # expert capacity
        capacity = (
            self._compute_multiplier_capacity(num_tokens)
            if self.expert_capacity is None
            else self.expert_capacity
        )

        # logits are shape (num_tokens, num_experts)
        # probs and indices are shape (num_experts, expert_capacity)
        router_logits, router_probs, expert_probs, expert_indices = self.router(
            x_flat,
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

            # compute expert output for all tokens
            # no scaling by routing probability
            shared_output = expert(x_flat)

            # accumulate expert output
            output += shared_output

        output = output.view(batch_size, seq_len, d_model)
        return output, (router_logits, router_probs, expert_probs, expert_indices)
