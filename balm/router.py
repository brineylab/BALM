# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

__all__ = ["TopKRouter", "ExpertChoiceRouter"]


class TopKRouter(nn.Module):
    """
    Router that implements the "token choice of top-k experts" strategy. For example, if k=1, this
    replicates the top-1 routing strategy introduced in the `Switch Transformers`_ paper.
    Alternatively, if k=2, this replicates the top-2 routing strategy introduced in the `GShard`_
    paper. Tokens are routed to their expert of choice until the expert's `expert_capacity` is
    reached. Shared experts, which process all tokens, are implemented as described in the
    `DeepSeqMoE`_ paper. Higher router precision and input jitter, introduced in the `Switch 
    Transformers`_ paper, is modeled on the `Mixtral`_ implementation.

    .. note::
        There is no guarantee that each token will be processed by an expert,
        or that every expert will receive at least one token.

    If tokens are routed to an expert which is above capacity, they are not processed by any expert
    and their hidden states are passed to the subsequent layer unchanged.

    Parameters:
    -----------
    d_model: int
        Token embedding dimension
    num_experts: int
        Number of available experts
    router_bias: bool
        Whether to use bias
    router_dtype: torch.dtype
        Data type to use for softmax of the router.
    router_jitter: float
        Jitter to apply to inputs.

    Input shape: (num_tokens, d_model)

    Returns:
    --------
    logits: (num_tokens, num_experts) routing scores
    probs: (num_tokens, num_experts) routing probabilities
    expert_probs: (num_experts, expert_capacity) expert-selected token probabilities
    expert_indices: (num_experts, expert_capacity) selected token indices

    .. _Switch Transformers:
        https://arxiv.org/abs/2101.03961

    .. _GShard:
        https://arxiv.org/abs/2006.16668

    .. _DeepSeqMoE:
        https://arxiv.org/abs/2401.06066
    
    .. _Mixtral HuggingFace Code:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L86
    """

    def __init__(
        self, 
        d_model: int, 
        num_experts: int,
        router_bias: bool,
        router_dtype: torch.dtype,
        router_jitter: float,
    ):
        super().__init__()
        self.router_dtype = router_dtype
        self.router_jitter = router_jitter

        self.linear = nn.Linear(d_model, num_experts, bias=router_bias)

    def forward(
        self, 
        x: torch.Tensor, 
        k: int, 
        expert_capacity: int
    ):
        """
        x: (num_tokens, d_model)
        k: how many experts each token chooses
        expert_capacity: how many tokens each expert keeps
        """
        num_tokens = x.size(0)
        num_experts = self.linear.out_features
        k = min(k, num_experts)

        # add jitter if training
        if self.training and self.router_jitter > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.router_jitter, 1.0 + self.router_jitter)

        # compute routing logits and probs ==> (num_tokens, num_experts)
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1, dtype=self.router_dtype) # softmax in higher precision (fp32 for stability)

        # select top-k experts for each token ==> (num_tokens, k)
        topk_probs, topk_expert_ids = torch.topk(probs, k, dim=-1)

        # convert back to original dtype
        topk_probs = topk_probs.to(x.dtype)

        # flatten top-k expert IDs and probs ==> (num_tokens * k)
        flat_expert_ids = topk_expert_ids.reshape(-1)
        flat_probs = topk_probs.reshape(-1)

        # we also need the original token ids that correspond to each slot
        # in flattened form: e.g. 0..(num_tokens-1) repeated k times ==> (num_tokens * k)
        flat_token_ids = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(num_tokens, k)
            .reshape(-1)
        )

        # for each expert, pick up to 'expert_capacity' tokens
        # in order of highest expert affiniy
        expert_probs = torch.zeros(
            (num_experts, expert_capacity), dtype=probs.dtype, device=probs.device
        )
        expert_indices = torch.full(
            (num_experts, expert_capacity),
            fill_value=-1,
            dtype=torch.long,
            device=probs.device,
        )

        # loop over experts, grouping slots that picked each expert.
        for e in range(num_experts):
            mask = flat_expert_ids == e
            chosen_token_ids = flat_token_ids[mask]
            chosen_probs = flat_probs[mask]

            # sort by expert affinity
            if chosen_token_ids.numel() > 0:
                sorted_probs, sorted_idx = torch.sort(chosen_probs, descending=True)
                # take up to `expert_capacity` tokens per expert
                keep = min(expert_capacity, sorted_probs.size(0))
                expert_probs[e, :keep] = sorted_probs[:keep]
                expert_indices[e, :keep] = chosen_token_ids[sorted_idx[:keep]]

        return logits, probs, expert_probs, expert_indices


class ExpertChoiceRouter(nn.Module):
    """
    This router uses the "expert choice of top-k tokens" strategy, as originally described
    in the `Mixture-of-Experts with Expert Choice Routing`_ paper. This automatically
    balances the number of tokens processed by each expert, and eliminates the
    need for an auxiliary (load-balancing) router loss. Higher router precision and input jitter, 
    introduced in the `Switch Transformers`_ paper, is modeled on the `Mixtral`_ implementation.

    .. note::
        There is no guarantee that each token will be processed by an expert. In fact,
        one of the primary benefits of expert choice routing is thought to be their
        ability to heterogeneously devote computation to a subset of highly complex/difficult
        tokens.

    If tokens are not selected by an expert, their hidden states are passed to the
    subsequent layer unchanged.

    Parameters:
    -----------
    d_model: int
        Token embedding dimension
    num_experts: int
        Number of available experts
    router_bias: bool
        Whether to use bias
    router_dtype: torch.dtype
        Data type to use for softmax of the router.
    router_jitter: float
        Jitter to apply to inputs of the router.

    Input shape: (num_tokens, d_model)

    Returns:
    --------
    logits: (num_tokens, num_experts) routing scores
    probs: (num_tokens, num_experts) routing probabilities
    expert_probs: (num_experts, expert_capacity) expert-selected token probabilities
    expert_indices: (num_experts, expert_capacity) selected token indices

    .. _Mixture-of-Experts with Expert Choice Routing:
        https://arxiv.org/abs/2202.09368

    .. _Switch Transformers:
        https://arxiv.org/abs/2101.03961
    
    .. _Mixtral HuggingFace Code:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L86

    """

    def __init__(
        self, 
        d_model: int, 
        num_experts: int,
        router_bias: bool,
        router_dtype: torch.dtype,
        router_jitter: float
    ):
        super().__init__()
        self.router_dtype = router_dtype
        self.router_jitter = router_jitter

        self.linear = nn.Linear(d_model, num_experts, bias=router_bias)

    def forward(
        self, 
        x: torch.Tensor, 
        k: int, 
        expert_capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # add jitter if training
        if self.training and self.router_jitter > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.router_jitter, 1.0 + self.router_jitter)
        
        # compute routing logits ==> (num_tokens, num_experts)
        logits = self.linear(x)

        # softmax in higher precision (fp32 for stability)
        probs = F.softmax(logits, dim=0, dtype=self.router_dtype) 

        # select tokens for each expert
        expert_probs, expert_indices = torch.topk(probs, k=expert_capacity, dim=0)

        # convert back to original dtype
        expert_probs = expert_probs.to(x.dtype)

        return logits, probs, expert_probs.T, expert_indices.T
