# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Tuple, Optional

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
        **kwargs,
    ):
        super().__init__()
        self.num_experts = num_experts
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

    def _assign_tokens_to_experts(
        self,
        flat_expert_ids: torch.Tensor,
        flat_token_ids: torch.Tensor,
        flat_probs: torch.Tensor,
        flat_unnorm_probs: torch.Tensor,
        num_tokens: int,
        expert_capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assigns tokens to experts based on top-k or top-p routing outputs.

        Returns:
        --------
        expert_probs : torch.Tensor
            Tensor of shape (num_experts, expert_capacity) containing the selected normalized probabilities.
        expert_indices : torch.Tensor
            Tensor of shape (num_experts, expert_capacity) containing token indices assigned to each expert.
        """
        # handle no expert capacity
        # max number of tokens one expert could take is the num_tokens
        if expert_capacity == -1:
            expert_capacity = num_tokens

        # initalize empty tensors for results
        expert_probs = torch.zeros(
            (self.num_experts, expert_capacity), dtype=dtype, device=device
        )
        expert_indices = torch.full(
            (self.num_experts, expert_capacity),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        # for each expert, pick up to 'expert_capacity' tokens in order of highest expert affinity
        # loop over experts, grouping slots that picked each expert
        for e in range(self.num_experts):
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

        return expert_probs, expert_indices


class TopKRouter(BaseRouter):
    """
    Router that implements the "token choice of top-k experts" strategy.

    If k=1, this matches the `Switch Transformers`_ top-1 routing.
    If k=2, it matches the `GShard`_ top-2 strategy.
    Shared expert routing is handled as in `DeepSeqMoE`_.

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

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        router_bias: bool,
        router_dtype: str,
        k: int,
        **kwargs,
    ):
        self.k = k
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            router_bias=router_bias,
            router_dtype=router_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_capacity: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Shape: `(num_tokens, d_model)`. Input token representations.
        expert_capacity : int
            Maximum number of tokens each expert can process.
        attention_mask : Optional[torch.Tensor]
            Shape: `(num_tokens)`. Flattened boolean mask for pad tokens.

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
        k = min(self.k, self.num_experts)

        # compute routing logits and probs ==> (num_tokens, num_experts)
        logits = self.linear(x)

        # mask padding tokens
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
            logits[pad_mask] = float(-1e9)

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

        # reshape indices and probs
        expert_probs, expert_indices = self._assign_tokens_to_experts(
            flat_expert_ids=flat_expert_ids,
            flat_token_ids=flat_token_ids,
            flat_probs=flat_probs,
            flat_unnorm_probs=flat_unnorm_probs,
            num_tokens=num_tokens,
            expert_capacity=expert_capacity,
            device=probs.device,
            dtype=probs.dtype,
        )

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
        top_p_threshold: float,
        **kwargs,
    ):
        self.threshold = top_p_threshold
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            router_bias=router_bias,
            router_dtype=router_dtype,
        )

    def forward(self, x: torch.Tensor, expert_capacity: int):
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

        num_tokens = x.size(0)

        # compute routing logits and probs ==> (num_tokens, num_experts)
        logits = self.linear(x)

        # softmax in higher precision (fp32 for stability)
        probs = F.softmax(logits, dim=-1, dtype=self.router_dtype)

        # sort probs
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # cumulative probs
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask
        mask = cumulative_probs <= self.threshold
        mask[..., 0] = (
            1  # always keep the first expert (accounts for tokens where highest prob is > threshold)
        )

        # apply mask
        selected_expert_ids = sorted_indices[mask]
        selected_probs = sorted_probs[mask]
        selected_token_ids = torch.nonzero(mask, as_tuple=True)[0]

        # reshape indices and probs
        expert_probs, expert_indices = self._assign_tokens_to_experts(
            flat_expert_ids=selected_expert_ids,
            flat_token_ids=selected_token_ids,
            flat_probs=selected_probs,
            flat_unnorm_probs=selected_probs,
            num_tokens=num_tokens,
            expert_capacity=expert_capacity,
            device=probs.device,
            dtype=probs.dtype,
        )

        return logits, probs, expert_probs, expert_indices


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

    def forward(self, x: torch.Tensor, expert_capacity: int):
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
