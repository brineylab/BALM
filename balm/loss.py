# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
from torch.nn import functional as F

__all__ = ["router_z_loss", "router_load_balancing_loss"]


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the router z-loss.

    The router z-loss was introduced in `Designing Effective Sparse Expert Models`_.
    It encourages router logits to remain small in an effort to improve stability.

    Parameters
    ----------
    router_logits : float
        Input logits of shape [batch_size * sequence_length, num_experts]

    Returns
    -------
    torch.Tensor
        The z-loss for the router

    .. _Designing Effective Sparse Expert Models:
        https://arxiv.org/abs/2202.08906
    """
    num_tokens, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / num_tokens


def router_load_balancing_loss(
    router_probs: torch.Tensor,
    k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the auxiliary load balancing loss.

    See the `Switch Transformer paper`_ for more details. This function
    implements the loss function presented in equations (4) - (6) of the paper.
    It aims at penalizing cases where the routing between experts is too unbalanced.
    Calculates the loss prior to dropping overflowed tokens, as demonstrated in figure
    (15) of the paper.

    Differing from the original `Switch Transformer paper`_, this is a more general
    implementation that supports both K=1 (top-1 routing) and K>1 (top-k routing):
      - for K=1 (top-1 routing), this reduces to the Switch Transformer load balancing loss.
      - for K>1 (top-k routing), this correctly accounts for multiple experts chosen per token.
    
    Implementation is slighlty modified from the `Mixtral implementation`_.

    Parameters
    ----------
    router_probs : torch.Tensor
        Concatenated router probs for all sparse layers.
        Shape: [num_tokens, num_experts], where:
            # num_tokens = (batch_size * seq_len * num_sparse_layers)
    
    k: int
        Number of experts each token is routed to (for topK routing).
    
    attention_mask: torch.Tensor, optional (default=None)
        Attention mask
        Shape: [batch_size, seq_len]

    Returns
    -------
    torch.Tensor
        The auxiliary load balancing loss for the router.

    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961

    .. _Mixtral implementation:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L851
    """
    num_tokens, num_experts = router_probs.shape
    device = router_probs.device
    
    # apply top-k across probs for all layers ==> (num_tokens, k)
    _, selected_experts = torch.topk(router_probs, k, dim=-1)

    # generate expert mask ==> (num_tokens, k, num_experts)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # factor attention_mask
    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts ==> (k, num_experts)
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts ==> (num_experts)
        router_prob_per_expert = torch.mean(router_probs, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = num_tokens // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, k, num_experts))
            .reshape(-1, k, num_experts)
            .to(device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(router_probs * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts
    return overall_loss
