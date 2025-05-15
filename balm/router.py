# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TopKRouter", "TopPRouter", "ExpertChoiceRouter"]


class BaseRouter(nn.Module):
    """
    Base class for MoE routers, handling initialization and dtype conversion.

    Attributes:
    -----------
    d_model: int
        Token embedding dimension
    num_experts: int
        Number of available experts
    router_bias: bool
        Whether to use bias
    router_dtype: torch.dtype
        Data type to use for softmax of the router.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        router_bias: bool,
        router_dtype: str,
    ):
        super().__init__()
        self.router_dtype = self._str_to_dtype(router_dtype)
        self.linear = nn.Linear(d_model, num_experts, bias=router_bias)

    def _str_to_dtype(self, dtype_str: str) -> torch.dtype:
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if dtype_str not in dtype_mapping:
            raise ValueError(
                f"Invalid dtype string: {dtype_str}. Choose from {list(dtype_mapping.keys())}"
            )

        return dtype_mapping[dtype_str]


class TopKRouter(BaseRouter):
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


    .. _Switch Transformers:
        https://arxiv.org/abs/2101.03961

    .. _GShard:
        https://arxiv.org/abs/2006.16668

    .. _DeepSeqMoE:
        https://arxiv.org/abs/2401.06066

    """

    def forward(self, x: torch.Tensor, k: int, expert_capacity: int):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Shape: `(num_tokens, d_model)`. Input token representations.
        k : int
            Maximum number of experts each token can choose.
        expert_capacity : int
            Maximum number of tokens each expert can process.

        Returns:
        --------
        logits : torch.Tensor
            Raw routing scores `(num_tokens, num_experts)`.
        probs : torch.Tensor
            Routing probabilities `(num_tokens, num_experts)`.
        expert_probs : torch.Tensor
            Selected token probabilities for each expert `(num_experts, expert_capacity)`.
        expert_indices : torch.Tensor
            Token indices assigned to each expert `(num_experts, expert_capacity)`.
        """

        num_tokens = x.size(0)
        num_experts = self.linear.out_features
        k = min(k, num_experts)

        # handle no expert capacity
        # max number of tokens one expert could take is the num_tokens
        if expert_capacity == -1:
            expert_capacity = num_tokens

        # compute routing logits and probs ==> (num_tokens, num_experts)
        logits = self.linear(x)

        # softmax in higher precision (fp32 for stability)
        probs = F.softmax(logits, dim=-1, dtype=self.router_dtype)

        # select top-k experts for each token ==> (num_tokens, k)
        topk_probs, topk_expert_ids = torch.topk(probs, k, dim=-1, sorted=False)

        # convert back to original dtype
        topk_probs = topk_probs.to(x.dtype)

        # normalize probs to sum to 1
        # but save unnormalized probs for sorting
        unnorm_topk_probs = topk_probs.clone().detach()
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)

        # flatten top-k expert IDs and probs ==> (num_tokens * k)
        flat_expert_ids, flat_unnorm_probs, flat_probs = (
            topk_expert_ids.flatten(),
            unnorm_topk_probs.flatten(),
            topk_probs.flatten(),
        )

        # we also need the original token ids that correspond to each slot
        # in flattened form: e.g. 0..(num_tokens-1) repeated k times ==> (num_tokens * k)
        flat_token_ids = torch.arange(num_tokens * k, device=x.device) // k

        # initalize empty tensors for results
        expert_probs = torch.zeros(
            (num_experts, expert_capacity), dtype=probs.dtype, device=probs.device
        )
        expert_indices = torch.full(
            (num_experts, expert_capacity),
            fill_value=-1,
            dtype=torch.long,
            device=probs.device,
        )

        # for each expert, pick up to 'expert_capacity' tokens in order of highest expert affinity
        # loop over experts, grouping slots that picked each expert
        for e in range(num_experts):
            # select token_ids and probs for expert e only
            mask = flat_expert_ids == e
            chosen_token_ids = flat_token_ids[mask]
            chosen_unnorm_probs = flat_unnorm_probs[mask]
            chosen_probs = flat_probs[mask]

            num_chosen_tokens = chosen_token_ids.numel()

            # sort by expert affinity
            if num_chosen_tokens > 0:

                # keep highest unnormalized probs, up to `expert_capacity` tokens per expert
                keep = min(expert_capacity, num_chosen_tokens)
                _, sorted_idx = torch.topk(chosen_unnorm_probs, keep, sorted=False)

                # filter for selected idxs
                expert_probs[e, :keep] = chosen_probs[
                    sorted_idx
                ]  # use normalized probs for weighting expert outputs
                expert_indices[e, :keep] = chosen_token_ids[sorted_idx]

        return logits, probs, expert_probs, expert_indices


class TopPRouter(BaseRouter):
    """
    Router that implements the "Top-P" strategy, introduced in the `Dynamic MoE Paper`_.

    .. _Dynamic MoE Paper:
        https://arxiv.org/abs/2403.07652

    .. _Dynamic MoE Code:
        https://github.com/ZhenweiAn/Dynamic_MoE/blob/main/modeling/modeling_moe.py

    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        router_bias: bool,
        router_dtype: str,
        top_p_threshold: Optional[float] = 0.7,
    ):
        self.threshold = top_p_threshold
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            router_bias=router_bias,
            router_dtype=router_dtype,
        )

    def forward(self, x: torch.Tensor, k: int, expert_capacity: int):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Shape: `(num_tokens, d_model)`. Input token representations.
        k : int
            Maximum number of experts each token can choose.
        expert_capacity : int
            Maximum number of tokens each expert can process.

        Returns:
        --------
        logits : torch.Tensor
            Raw routing scores `(num_tokens, num_experts)`.
        probs : torch.Tensor
            Routing probabilities `(num_tokens, num_experts)`.
        expert_probs : torch.Tensor
            Selected token probabilities for each expert `(num_experts, expert_capacity)`.
        expert_indices : torch.Tensor
            Token indices assigned to each expert `(num_experts, expert_capacity)`.
        """

        num_tokens = x.size(0)
        num_experts = self.linear.out_features
        k = min(k, num_experts)

        torch.set_printoptions(profile="full")

        # compute routing logits and probs ==> (num_tokens, num_experts)
        logits = self.linear(x)

        # softmax in higher precision (fp32 for stability)
        probs = F.softmax(logits, dim=-1, dtype=self.router_dtype)

        # sort probs
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # cumulative probs
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.threshold

        # find threshold indices
        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(
            threshold_indices, num_classes=sorted_indices.size(-1)
        ).bool()

        # apply masks ==> (num_tokens, num_experts)
        mask = mask & ~threshold_mask
        sorted_indices = torch.where(mask, -1, sorted_indices)
        sorted_probs = torch.where(mask, 0.0, sorted_probs)

        # reshape indices and probs
        raise Exception()

        return logits, probs, sorted_probs, sorted_indices


class ExpertChoiceRouter(BaseRouter):
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

    .. _Mixture-of-Experts with Expert Choice Routing:
        https://arxiv.org/abs/2202.09368

    """

    def forward(
        self, x: torch.Tensor, expert_capacity: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
        -----------
        x : torch.Tensor
            Shape: `(num_tokens, d_model)`. Input token representations.
        expert_capacity : int
            Maximum number of tokens each expert can process.

        Returns:
        --------
        logits : torch.Tensor
            Raw routing scores `(num_tokens, num_experts)`.
        probs : torch.Tensor
            Routing probabilities `(num_tokens, num_experts)`.
        expert_probs : torch.Tensor
            Selected token probabilities for each expert `(num_experts, expert_capacity)`.
        expert_indices : torch.Tensor
            Token indices assigned to each expert `(num_experts, expert_capacity)`.
        """

        # compute routing logits ==> (num_tokens, num_experts)
        logits = self.linear(x)

        # softmax in higher precision (fp32 for stability)
        probs = F.softmax(logits, dim=0, dtype=self.router_dtype)

        # select tokens for each expert
        expert_probs, expert_indices = torch.topk(probs, k=expert_capacity, dim=0)

        # convert back to original dtype
        expert_probs = expert_probs.to(x.dtype)

        return logits, probs, expert_probs.T, expert_indices.T
