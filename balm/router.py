# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

__all__ = ["TopKRouter", "ExpertChoiceRouter"]


# class TopKRouter(nn.Module):
#     """Learned token-to-expert routing module with top-k selection.

#     Args:
#         d_model: Token embedding dimension
#         num_experts: Number of available experts

#     Input shape: (num_tokens, d_model)
#     Outputs:
#         logits: (num_tokens, num_experts) routing scores
#         probs: (num_tokens, k) top-k routing probabilities
#         indices: (num_tokens, k) selected expert indices
#     """

#     def __init__(self, d_model: int, num_experts: int):
#         super().__init__()
#         self.linear = nn.Linear(d_model, num_experts)

#     def forward(
#         self, x: torch.Tensor, k: int
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         logits = self.linear(x)  # (num_tokens, num_experts)
#         probs = F.softmax(logits, dim=-1)
#         top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
#         return logits, top_k_probs, top_k_indices


# this one returns probs and indices as shape (num_experts, expert_capacity)
# which makes it compatible with the expert choice router
# class TopKRouter(nn.Module):
#     """Learned token-to-expert routing module with top-k selection and capacity filtering.
#     Args:
#         d_model: Token embedding dimension
#         num_experts: Number of available experts
#     Input shape: (num_tokens, d_model)
#     Outputs:
#         logits: (num_tokens, num_experts) routing scores
#         probs: (num_experts, expert_capacity) routing probabilities for selected tokens
#         indices: (num_experts, expert_capacity) indices of selected tokens
#     """

#     def __init__(self, d_model: int, num_experts: int):
#         super().__init__()
#         self.linear = nn.Linear(d_model, num_experts)

#     def forward(
#         self, x: torch.Tensor, k: int, expert_capacity: int
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         logits = self.linear(x)  # (num_tokens, num_experts)
#         probs = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)
#         num_tokens = x.size(0)
#         num_experts = self.linear.out_features

#         expert_probs = torch.zeros(
#             (num_experts, expert_capacity), dtype=probs.dtype, device=probs.device
#         )
#         expert_indices = torch.zeros(
#             (num_experts, expert_capacity),
#             dtype=torch.long,
#             device=indices.device
#             if (indices := torch.empty(0)).device != torch.device("cpu")
#             else torch.device("cpu"),
#         )

#         for expert_idx in range(num_experts):
#             expert_logits = logits[:, expert_idx]  # (num_tokens,)
#             sorted_logits, sorted_token_indices = torch.sort(
#                 expert_logits, descending=True
#             )
#             num_keep = min(
#                 expert_capacity, num_tokens
#             )  # handle cases where num_tokens < expert_capacity
#             if num_keep > 0:
#                 selected_token_indices = sorted_token_indices[:num_keep]
#                 selected_probs = probs[selected_token_indices, expert_idx]
#                 expert_probs[expert_idx, :num_keep] = selected_probs
#                 expert_indices[expert_idx, :num_keep] = selected_token_indices

#         return logits, expert_probs, expert_indices


# class TopKRouter(nn.Module):
#     """Learned token-to-expert routing module with top-k selection and capacity filtering.

#     Args:
#         d_model: Token embedding dimension
#         num_experts: Number of available experts

#     Input shape: (num_tokens, d_model)
#     Outputs:
#         logits: (num_tokens, num_experts) routing scores
#         probs: (num_experts, expert_capacity) routing probabilities for selected tokens
#         indices: (num_experts, expert_capacity) indices of selected tokens
#     """

#     def __init__(self, d_model: int, num_experts: int):
#         super().__init__()
#         self.linear = nn.Linear(d_model, num_experts)

#     def forward(
#         self, x: torch.Tensor, k: int, expert_capacity: int
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         num_tokens = x.size(0)
#         num_experts = self.linear.out_features

#         # routing logits and probabilities
#         logits = self.linear(x)  # ==> (num_tokens, num_experts)
#         probs = F.softmax(logits, dim=-1)  # ==> (num_tokens, num_experts)

#         # results tensors
#         expert_probs = torch.zeros(
#             (num_experts, expert_capacity), dtype=probs.dtype, device=probs.device
#         )
#         expert_indices = (
#             torch.zeros(
#                 (num_experts, expert_capacity), dtype=torch.long, device=probs.device
#             )
#             - 1  # -1 will later be filtered out as "empty slots" in an undersubscribed expert
#         )

#         # top-k experts selection
#         token_to_experts = torch.zeros(
#             (num_tokens, num_experts), dtype=torch.bool, device=probs.device
#         )
#         topk_probs, topk_indices = torch.topk(probs, k=min(k, num_experts), dim=1)

#         # map each token's top-k experts
#         for token_idx in range(num_tokens):
#             for expert_idx in topk_indices[token_idx]:
#                 token_to_experts[token_idx, expert_idx] = True

#         # capacity limiting per expert
#         for expert_idx in range(num_experts):
#             # get all tokens that selected this expert in their top-k
#             token_indices = torch.where(token_to_experts[:, expert_idx])[0]

#             if len(token_indices) > 0:
#                 # get the corresponding probabilities
#                 token_probs = probs[token_indices, expert_idx]

#                 # sort by probability (high to low)
#                 sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)

#                 # limit to expert capacity
#                 num_to_keep = min(expert_capacity, len(sorted_indices))

#                 if num_to_keep > 0:
#                     # select the tokens with highest probability for this expert
#                     selected_indices = sorted_indices[:num_to_keep]
#                     selected_token_indices = token_indices[selected_indices]
#                     selected_probs = sorted_probs[:num_to_keep]

#                     expert_probs[expert_idx, :num_to_keep] = selected_probs
#                     expert_indices[expert_idx, :num_to_keep] = selected_token_indices

#         return logits, probs, expert_probs, expert_indices


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


#


# class RouterBase(nn.Module):
#     """
#     Base class for routers.

#     Parameters:
#     -----------
#     config : PretrainedConfig
#         Configuration object containing router parameters.

#     """

#     def __init__(self, config: PretrainedConfig):
#         super().__init__()
#         self.config = config
#         self.dtype = getattr(torch, self.config.router_dtype)

#         # compute expert capacity
#         if self.config.expert_capacity_type == "multiplier":
#             self.expert_capacity = int(
#                 self.config.expert_capacity
#                 * self.config.hidden_size
#                 / self.config.num_experts
#             )
#         else:
#             self.expert_capacity = self.config.expert_capacity

#         # create router
#         self.classifier = nn.Linear(
#             self.config.hidden_size,
#             self.config.num_experts - self.config.num_shared_experts,
#             bias=self.config.router_bias,
#             dtype=self.dtype,
#         )

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         raise NotImplementedError

#     def _compute_router_probabilities(
#         self,
#         x: torch.Tensor,
#         dim: int = -1,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes router probabilities from input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, hidden_dim) from which
#             router probabilities are computed.

#         dim : int, optional
#             Dimension along which to compute the softmax. The default is -1, which corresponds
#             to token-choice routing. For expert choice routing, this should be -2.

#         Returns:
#         --------
#         router_probabilities : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) corresponding to
#             the probabilities for each token and expert. Used for routing tokens to experts.

#         router_logits : torch.Tensor
#             Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding
#             to raw router logits. This is used for computing router z-loss.
#         """
#         # float32 is typically used to ensure stability, see https://arxiv.org/abs/2101.03961.
#         self.input_dtype = x.dtype
#         x = x.to(self.dtype)
#         if self.config.router_jitter > 0:
#             jitter = torch.empty_like(x).uniform(
#                 1.0 - self.config.router_jitter, 1.0 + self.config.router_jitter
#             )
#             x = x * jitter
#         logits = self.classifier(x)  # -> (batch, seq_len, num_experts)
#         probabilities = F.softmax(logits, dim=dim, dtype=self.dtype).to(
#             self.input_dtype
#         )
#         return probabilities, logits


# class TopKRouter(RouterBase):
#     """
#     This router uses the "token choice of top-k experts" strategy. For example, if k=1, this
#     replicates the top-1 routing strategy introduced in the `Switch Transformers`_ paper.
#     Alternatively, if k=2, this replicates the top-2 routing strategy introduced in the `GShard`_
#     paper. Tokens are routed to their expert of choice until the expert's `expert_capacity` is
#     reached. Shared experts, which process all tokens, are implemented as described in the
#     `DeepSeqMoE`_ paper.

#     .. note::
#         There is no guarantee that each token will be processed by an expert,
#         or that every expert will receive at least one token.

#     If tokens are routed to an expert which is above capacity, they are not processed by any expert
#     and their hidden states are passed to the subsequent layer unchanged.

#     Parameters:
#     -----------
#     config : PretrainedConfig
#         Configuration object containing router parameters.


#     .. _Switch Transformers:
#         https://arxiv.org/abs/2101.03961

#     .. _GShard:
#         https://arxiv.org/abs/2006.16668

#     .. _DeepSeqMoE:
#         https://arxiv.org/abs/2401.06066
#     """

#     def __init__(self, config: PretrainedConfig):
#         super().__init__(config)

#     def forward(
#         self, x: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Token choice of top-k experts, with optional shared experts processing all tokens.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         expert_mask : torch.Tensor
#             Binary mask tensor of shape (batch_size, sequence_length, num_experts)
#             indicating which experts the token should be routed to (including shared experts).

#         router_probs : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router probabilities.

#         router_logits : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router logits.
#         """
#         num_routable_experts = self.config.num_experts - self.config.num_shared_experts
#         router_probs, router_logits = self._compute_router_probabilities(x)
#         _, topk_indices = torch.topk(
#             router_probs, k=self.config.num_experts_per_tok, dim=-1
#         )
#         expert_mask = F.one_hot(topk_indices, num_classes=num_routable_experts).sum(
#             dim=-2
#         )

#         # mask tokens if their desired experts are above capacity
#         token_priority = torch.cumsum(expert_mask, dim=-2)
#         expert_capacity_mask = token_priority <= self.expert_capacity
#         expert_mask = expert_mask * expert_capacity_mask

#         # shared experts
#         if self.config.num_shared_experts > 0:
#             # include shared experts in the expert mask (first N experts are shared experts)
#             shared_expert_mask = torch.ones_like(
#                 router_probs[..., : self.config.num_shared_experts]
#             )
#             expert_mask = torch.cat((shared_expert_mask, expert_mask), dim=-1)
#             # add shared experts to router probs
#             shared_expert_probs = torch.ones_like(
#                 router_probs[..., : self.config.num_shared_experts]
#             )
#             router_probs = torch.cat((shared_expert_probs, router_probs), dim=-1)

#         return expert_mask, router_probs, router_logits


# class ExpertChoiceRouter(RouterBase):
#     """
#     This router uses the "expert choice of top-k tokens" strategy, as originally described
#     in the `Mixture-of-Experts with Expert Choice Routing`_ paper. This automatically
#     balances the number of tokens processed by each expert, and eliminates the
#     need for an auxiliary (load-balancing) router loss.

#     .. note::
#         There is no guarantee that each token will be processed by an expert. In fact,
#         one of the primary benefits of expert choice routing is thought to be their
#         ability to heterogeneously devote computation to a subset of highly complex/difficult
#         tokens.

#     If tokens are not selected by an expert, their hidden states are passed to the
#     subsequent layer unchanged.

#     Parameters:
#     -----------
#     config : PretrainedConfig
#         Configuration object containing router parameters.

#     .. _Mixture-of-Experts with Expert Choice Routing:
#         https://arxiv.org/abs/2202.09368
#     """

#     def __init__(self, config: PretrainedConfig):
#         super().__init__(config)

#     def forward(
#         self, x: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Expert choice of top-k tokens, with optional shared experts that process all tokens.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         expert_mask : torch.Tensor
#             Binary mask tensor of shape (batch_size, sequence_length, num_experts) indicating
#             which tokens are selected for each expert and which are processed by shared experts.

#         router_probs : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router probabilities.

#         router_logits : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router logits.
#         """
#         router_probs, router_logits = self._compute_router_probabilities(x, dim=-2)
#         expert_mask = torch.zeros_like(router_probs)

#         # select top-k tokens for each routable (non-shared) expert
#         for i in range(self.config.num_experts - self.config.num_shared_experts):
#             _, top_k_indices = torch.topk(
#                 router_probs[..., i], k=self.expert_capacity, dim=1
#             )
#             expert_mask[:, :, i].scatter_(1, top_k_indices, 1)

#         # shared experts
#         if self.config.num_shared_experts > 0:
#             # include shared experts in the expert mask (first N experts are shared experts)
#             shared_expert_mask = torch.ones_like(
#                 router_probs[..., : self.config.num_shared_experts]
#             )
#             expert_mask = torch.cat((shared_expert_mask, expert_mask), dim=-1)
#             # add shared experts to router probs
#             shared_expert_probs = torch.ones_like(
#                 router_probs[..., : self.config.num_shared_experts]
#             )
#             router_probs = torch.cat((shared_expert_probs, router_probs), dim=-1)

#         return expert_mask, router_probs, router_logits


# class RouterBase(nn.Module):
#     """
#     Base class for routers.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_experts: int,
#         expert_capacity: int,
#         dtype: str = "float32",
#         bias: bool = False,
#         jitter: float = 0.0,
#         num_routable_experts: Optional[int] = None,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_experts = num_experts
#         self.expert_capacity = expert_capacity
#         self.dtype = getattr(torch, dtype)
#         self.bias = bias
#         self.jitter = jitter
#         self.classifier = nn.Linear(
#             self.embed_dim,
#             num_routable_experts
#             if num_routable_experts is not None
#             else self.num_experts,
#             bias=self.bias,
#             dtype=self.dtype,
#         )

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         raise NotImplementedError

#     def _compute_router_probabilities(
#         self,
#         x: torch.Tensor,
#         dim: int = -1,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes router probabilities from input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, hidden_dim) from which
#             router probabilities are computed.

#         dim : int, optional
#             Dimension along which to compute the softmax. The default is -1, which corresponds
#             to token-choice routing. For expert choice routing, this should be -2.

#         Returns:
#         --------
#         router_probabilities : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) corresponding to
#             the probabilities for each token and expert. Used for routing tokens to experts.

#         router_logits : torch.Tensor
#             Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding
#             to raw router logits. This is used for computing router z-loss.
#         """
#         # float32 is used to ensure stability. See https://arxiv.org/abs/2101.03961.
#         self.input_dtype = x.dtype
#         x = x.to(self.dtype)
#         if self.jitter > 0:
#             jitter = torch.empty_like(x).uniform(1.0 - self.jitter, 1.0 + self.jitter)
#             x = x * jitter
#             # x *= torch.empty_like(x).uniform_(1.0 - self.jitter, 1.0 + self.jitter)
#         logits = self.classifier(x)  # (batch, seq_len, num_experts)
#         probabilities = F.softmax(logits, dim=dim, dtype=self.dtype).to(
#             self.input_dtype
#         )
#         return probabilities, logits


# class TopKRouter(RouterBase):
#     """
#     This router uses the "token choice of top-k experts" strategy. For example, if k=1, this
#     replicates the top-1 routing strategy introduced in the `Switch Transformers`_ paper.
#     Alternatively, if k=2, this replicates the top-2 routing strategy introduced in the `GShard`_
#     paper. Tokens are routed to their expert of choice until the expert's `expert_capacity` is
#     reached. Shared experts, which process all tokens, are implemented as described in the
#     `DeepSeqMoE`_ paper.

#     .. note::
#         There is no guarantee that each token will be processed by an expert,
#         or that every expert will receive at least one token.

#     If tokens are routed to an expert which is above capacity, they are not processed by any expert
#     and their hidden states are passed to the subsequent layer unchanged.


#     Parameters:
#     -----------
#     embed_dim : int
#         Embedding dimension.

#     num_experts : int
#         Number of experts.

#     expert_capacity : int
#         Maximum number of tokens that can be routed to each expert.

#     top_k : int, optional
#         Number of top experts to route each token to. The default is 1.

#     num_shared_experts : int, optional
#         Number of shared experts that process all tokens. The default is 0.

#     send_bos_to_all_experts : bool, optional
#         Whether to send the BOS token to all experts. The default is ``False``.

#     dtype : str, optional
#         Data type to use for router probabilities. The default is "float32".

#     bias : bool, optional
#         Whether to add bias to the router classifier. The default is ``False``.

#     jitter : float, optional
#         Amount of jitter to add to the router probabilities. The default is ``0.0``.

#     ignore_padding_tokens : bool, optional
#         Whether to ignore padding tokens when computing router probabilities.
#         The default is ``True``.


#     .. _Switch Transformers:
#         https://arxiv.org/abs/2101.03961

#     .. _GShard:
#         https://arxiv.org/abs/2006.16668

#     .. _DeepSeqMoE:
#         https://arxiv.org/abs/2401.06066
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_experts: int,
#         expert_capacity: int,
#         top_k: int = 1,
#         num_shared_experts: int = 0,
#         # send_bos_to_all_experts: bool = False,
#         dtype: str = "float32",
#         bias: bool = False,
#         jitter: float = 0.0,
#         ignore_padding_tokens: bool = True,
#         **kwargs,
#     ):
#         super().__init__(
#             embed_dim=embed_dim,
#             num_experts=num_experts,
#             expert_capacity=expert_capacity,
#             dtype=dtype,
#             bias=bias,
#             jitter=jitter,
#             num_routable_experts=num_experts - num_shared_experts,
#         )
#         self.top_k = top_k
#         self.num_shared_experts = num_shared_experts
#         # self.send_bos_to_all_experts = send_bos_to_all_experts
#         self.ignore_padding_tokens = ignore_padding_tokens

#     def forward(
#         self, x: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Token choice of top-k experts, with optional shared experts processing all tokens.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         expert_mask : torch.Tensor
#             Binary mask tensor of shape (batch_size, sequence_length, num_experts)
#             indicating which experts the token should be routed to (including shared experts).

#         router_probabilities : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router probabilities.

#         router_logits : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router logits.
#         """
#         num_routable_experts = self.num_experts - self.num_shared_experts

#         # router
#         router_probs, router_logits = self._compute_router_probabilities(x)
#         _, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
#         expert_mask = F.one_hot(topk_indices, num_classes=num_routable_experts).sum(
#             dim=-2
#         )

#         # # adjust expert capacity if BOS is sent to all experts
#         # expert_capacity = self.expert_capacity
#         # if self.send_bos_to_all_experts:
#         #     expert_capacity -= 1
#         #     router_probs = router_probs[:, 1:, :]
#         #     expert_mask = expert_mask[:, 1:, :]

#         # mask tokens if their desired experts are above capacity
#         token_priority = torch.cumsum(expert_mask, dim=-2)
#         expert_capacity_mask = token_priority <= self.expert_capacity
#         expert_mask = expert_mask * expert_capacity_mask

#         # shared experts
#         if self.num_shared_experts > 0:
#             # include shared experts in the expert mask (first N experts are shared experts)
#             shared_expert_mask = torch.ones_like(
#                 router_probs[..., : self.num_shared_experts]
#             )
#             expert_mask = torch.cat((shared_expert_mask, expert_mask), dim=-1)
#             # add shared experts to router probs
#             shared_expert_probs = torch.ones_like(
#                 router_probs[..., : self.num_shared_experts]
#             )
#             router_probs = torch.cat((shared_expert_probs, router_probs), dim=-1)

#         # # send BOS token to all experts, if specified
#         # if self.send_bos_to_all_experts:
#         #     # add BOS token to expert mask
#         #     bos_mask = torch.ones_like(expert_mask[:, 0, :]).unsqueeze(1)
#         #     expert_mask = torch.cat((bos_mask, expert_mask), dim=1)
#         #     # add BOS token to router probs
#         #     # TODO: could try removing this so learned router probs propagate
#         #     bos_probs = torch.ones_like(router_probs[:, 0, :]).unsqueeze(1)
#         #     router_probs = torch.cat((bos_probs, router_probs), dim=1)

#         return expert_mask, router_probs, router_logits


# class ExpertChoiceRouter(RouterBase):
#     """
#     This router uses the "expert choice of top-k tokens" strategy, as originally described
#     in the `Mixture-of-Experts with Expert Choice Routing`_ paper. This automatically
#     balances the number of tokens processed by each expert, and eliminates the
#     need for an auxiliary (load-balancing) router loss.

#     .. note::
#         There is no guarantee that each token will be processed by an expert. In fact,
#         one of the primary benefits of expert choice routing is thought to be their
#         ability to heterogeneously devote computation to a subset of highly complex/difficult
#         tokens.

#     If tokens are not selected by an expert, their hidden states are passed to the
#     subsequent layer unchanged.

#     Parameters:
#     -----------
#     embed_dim : int
#         Embedding dimension.

#     num_experts : int
#         Number of experts.

#     expert_capacity : int
#         Maximum number of tokens that can be routed to each expert.

#     num_shared_experts : int, optional
#         Number of shared experts that process all tokens. The default is 0.

#     send_bos_to_all_experts : bool, optional
#         Whether to send the BOS token to all experts. The default is ``False``.

#     dtype : str, optional
#         Data type to use for router probabilities. The default is "float32".

#     bias : bool, optional
#         Whether to add bias to the router classifier. The default is ``False``.

#     jitter : float, optional
#         Amount of jitter to add to the router probabilities. The default is ``0.0``.

#     ignore_padding_tokens : bool, optional
#         Whether to ignore padding tokens when computing router probabilities.
#         The default is ``True``.

#     .. _Mixture-of-Experts with Expert Choice Routing:
#         https://arxiv.org/abs/2202.09368
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_experts: int,
#         expert_capacity: int,
#         num_shared_experts: int = 0,
#         # send_bos_to_all_experts: bool = False,
#         dtype: str = "float32",
#         bias: bool = False,
#         jitter: float = 0.0,
#         ignore_padding_tokens: bool = True,
#         **kwargs,
#     ):
#         super().__init__(
#             embed_dim=embed_dim,
#             num_experts=num_experts,
#             expert_capacity=expert_capacity,
#             dtype=dtype,
#             bias=bias,
#             jitter=jitter,
#             num_routable_experts=num_experts - num_shared_experts,
#         )
#         self.num_shared_experts = num_shared_experts
#         # self.send_bos_to_all_experts = send_bos_to_all_experts
#         self.ignore_padding_tokens = ignore_padding_tokens

#     def forward(
#         self, x: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Expert choice of top-k tokens, with optional shared experts that process all tokens.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         Returns:
#         --------
#         expert_mask : torch.Tensor
#             Binary mask tensor of shape (batch_size, sequence_length, num_experts) indicating
#             which tokens are selected for each expert and which are processed by shared experts.

#         router_probabilities : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router probabilities.

#         router_logits : torch.Tensor
#             Tensor of shape (batch_size, sequence_length, num_experts) containing
#             the router logits.
#         """
#         router_probs, router_logits = self._compute_router_probabilities(x, dim=-2)
#         expert_mask = torch.zeros_like(router_probs)

#         # # adjust expert capacity if BOS is sent to all experts
#         # expert_capacity = self.expert_capacity
#         # if self.send_bos_to_all_experts:
#         #     expert_capacity -= 1
#         #     router_probs = router_probs[:, 1:, :]
#         #     expert_mask = expert_mask[:, 1:, :]

#         # select top-k tokens for each expert
#         for i in range(self.num_experts - self.num_shared_experts):
#             _, top_k_indices = torch.topk(
#                 router_probs[..., i], k=self.expert_capacity, dim=1
#             )
#             # expert_mask[:, :, i] = expert_mask[:, :, i].scatter(1, top_k_indices, 1)
#             expert_mask[:, :, i].scatter_(1, top_k_indices, 1)

#         # shared experts
#         if self.num_shared_experts > 0:
#             # include shared experts in the expert mask (first N experts are shared experts)
#             shared_expert_mask = torch.ones_like(
#                 router_probs[..., : self.num_shared_experts]
#             )
#             expert_mask = torch.cat((shared_expert_mask, expert_mask), dim=-1)
#             # add shared experts to router probs
#             shared_expert_probs = torch.ones_like(
#                 router_probs[..., : self.num_shared_experts]
#             )
#             router_probs = torch.cat((shared_expert_probs, router_probs), dim=-1)

#         # # send BOS token to all experts, if specified
#         # if self.send_bos_to_all_experts:
#         #     # add BOS token to expert mask
#         #     bos_mask = torch.ones_like(expert_mask[:, 0, :]).unsqueeze(1)
#         #     expert_mask = torch.cat((bos_mask, expert_mask), dim=1)
#         #     # add BOS token to router probs
#         #     # TODO: could try removing this so learned router probs propagate
#         #     bos_probs = torch.ones_like(router_probs[:, 0, :]).unsqueeze(1)
#         #     router_probs = torch.cat((bos_probs, router_probs), dim=1)

#         return expert_mask, router_probs, router_logits
