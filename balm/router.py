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
    `DeepSeqMoE`_ paper.

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
    """

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor, k: int, expert_capacity: int):
        """
        x: (num_tokens, d_model)
        k: how many experts each token chooses
        expert_capacity: how many tokens each expert keeps
        """
        num_tokens = x.size(0)
        num_experts = self.linear.out_features
        k = min(k, num_experts)

        # compute routing logits and probs ==> (num_tokens, num_experts)
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)

        # select top-k experts for each token ==> (num_tokens, k)
        topk_probs, topk_expert_ids = torch.topk(probs, k, dim=-1)

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
    need for an auxiliary (load-balancing) router loss.

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

    Input shape: (num_tokens, d_model)

    Returns:
    --------
    logits: (num_tokens, num_experts) routing scores
    probs: (num_tokens, num_experts) routing probabilities
    expert_probs: (num_experts, expert_capacity) expert-selected token probabilities
    expert_indices: (num_experts, expert_capacity) selected token indices

    .. _Mixture-of-Experts with Expert Choice Routing:
        https://arxiv.org/abs/2202.09368

    """

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts)

    def forward(
        self, x: torch.Tensor, k: int, expert_capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(x)  # (num_tokens, num_experts)
        probs = F.softmax(logits, dim=0)
        expert_probs, expert_indices = torch.topk(probs, k=expert_capacity, dim=0)
        return logits, probs, expert_probs.T, expert_indices.T
