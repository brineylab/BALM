# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

__all__ = ["TopKRouter", "ExpertChoiceRouter"]

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
            raise ValueError(f"Invalid dtype string: {dtype_str}. Choose from {list(dtype_mapping.keys())}")
        
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
    def forward(
        self, 
        x: torch.Tensor, 
        padding_mask: torch.Tensor,
        k: int, 
        expert_capacity: int
    ):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Shape: `(num_tokens, d_model)`. Input token representations.
        padding_mask : Optional[torch.Tensor]
            Shape: `(batch_size, seq_len)`. Boolean mask for padded tokens (not yet applied).
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
        topk_probs, topk_expert_ids = torch.topk(probs, k, dim=-1)

        # convert back to original dtype
        topk_probs = topk_probs.to(x.dtype)

        # normalize probs to sum to 1
        # but save unnormalized probs for sorting
        unnorm_topk_probs = topk_probs.clone().detach()
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True) 

        # TODO
        # mask padding tokens
        # if padding_mask is not None:
        #     topk_expert_ids = topk_expert_ids[~padding_mask]
        #     topk_probs = topk_probs[~padding_mask]

        #     # generate token ids for below
        #     token_ids = torch.arange(num_tokens, device=x.device)[~padding_mask]
        #     num_tokens = len(token_ids)
        # else:
        #     # generate token ids for below
        #     token_ids = torch.arange(num_tokens, device=x.device)


        # flatten top-k expert IDs and probs ==> (num_tokens * k)
        flat_expert_ids, flat_unnorm_probs, flat_probs = (
            topk_expert_ids.reshape(-1), unnorm_topk_probs.reshape(-1), topk_probs.reshape(-1)
        )

        # we also need the original token ids that correspond to each slot
        # in flattened form: e.g. 0..(num_tokens-1) repeated k times ==> (num_tokens * k)
        flat_token_ids = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(num_tokens, k)
            .reshape(-1)
        )

        # initalize empty tensors for results
        expert_probs = torch.zeros(
            (num_experts, expert_capacity), dtype=probs.dtype, device=probs.device
        )
        expert_indices = torch.full(
            (num_experts, expert_capacity), fill_value=-1, dtype=torch.long, device=probs.device,
        )

        # for each expert, pick up to 'expert_capacity' tokens in order of highest expert affiniy
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
                _, sorted_idx = torch.topk(chosen_unnorm_probs, keep)

                # filter for selected idxs
                expert_probs[e, :keep] = chosen_probs[sorted_idx] # use normalized probs for weighting expert outputs
                expert_indices[e, :keep] = chosen_token_ids[sorted_idx]

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
    def forward(
        self, 
        x: torch.Tensor, 
        padding_mask: torch.Tensor,
        k: int, 
        expert_capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
        -----------
        x : torch.Tensor
            Shape: `(num_tokens, d_model)`. Input token representations.
        padding_mask : Optional[torch.Tensor]
            Shape: `(batch_size, seq_len)`. Boolean mask for padded tokens (not yet applied).
        k : int
            Unused by expert choice router. Present for compatibility with TopKRouter
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
