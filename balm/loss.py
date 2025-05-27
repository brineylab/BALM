# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, List

import torch

__all__ = ["router_z_loss", "router_load_balancing_loss", "router_p_penalty_loss"]


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the router z-loss.

    The router z-loss was introduced in `Designing Effective Sparse Expert Models`_.
    It encourages router logits to remain small in an effort to improve stability.

    Parameters
    ----------
    router_logits : torch.Tensor
        Input logits of shape [batch_size * sequence_length, num_experts].

    Returns
    -------
    torch.Tensor
        The z-loss for the router.

    References
    ----------
    .. _Designing Effective Sparse Expert Models:
        https://arxiv.org/abs/2202.08906
    """

    num_tokens, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / num_tokens


def router_load_balancing_loss(
    router_probs: torch.Tensor,
    router_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the auxiliary load balancing loss.

    See the `Switch Transformer paper`_ for more details. This function
    implements the loss function presented in equations (4) - (6) of the paper.
    It aims at penalizing cases where the routing between experts is too unbalanced.
    Calculates the loss prior to dropping overflowed tokens, as demonstrated in figure
    (15) of the paper.

    Differing from the original `Switch Transformer paper`_, this implementation supports
    K>1 (top-k routing) where each token chooses multiple experts

    Implementation is slightly modified from the `Mixtral implementation`_.

    Parameters
    ----------
    router_ids : torch.Tensor
        Concatenated router probs for all sparse layers. Shape is [num_tokens, k, num_experts],
        where num_tokens is (batch_size * seq_len * num_sparse_layers)
    k: int
        Number of experts each token is routed to (for topK routing).
    attention_mask: torch.Tensor, optional
        Attention mask of shape: [batch_size, seq_len].

    Returns
    -------
    torch.Tensor
        The auxiliary load balancing loss for the router.

    References
    ----------
    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961

    .. _Mixtral implementation:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L851
    """

    num_tokens, k, num_experts = router_ids.shape
    device = router_ids.device

    # factor attention_mask
    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts ==> (k, num_experts)
        tokens_per_expert = torch.mean(router_ids.float(), dim=0)

        # Compute the average probability of routing to these experts ==> (num_experts)
        router_prob_per_expert = torch.mean(router_probs, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = num_tokens // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of router_ids
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, k, num_experts))
            .reshape(-1, k, num_experts)
            .to(device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            router_ids.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            router_probs * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    return (
        torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts
    )


def router_p_penalty_loss(
    router_probs: torch.Tensor, k: int, expert_hidden_sizes: List
) -> torch.Tensor:
    """
    Computes the parameter penalty (P-Penalty) loss.

    See the `HMoE paper`_ for more details.

    Parameters
    ----------
    router_probs : torch.Tensor
        Concatenated router probs for all sparse layers. Shape is [num_tokens, num_experts],
        where num_tokens is (batch_size * seq_len * num_sparse_layers)
    k: int
        Number of experts each token is routed to (for topK routing).
    expert_hidden_sizes: torch.Tensor
        Hidden sizes of the experts.

    Returns
    -------
    torch.Tensor
        The p-penalty loss for the router.

    References
    ----------
    .. _HMoE paper:
        https://arxiv.org/abs/2408.10681
    """

    _, num_experts = router_probs.shape
    device = router_probs.device

    # apply top-k across probs for all layers ==> (num_tokens, k)
    _, selected_experts = torch.topk(router_probs, k, dim=-1)

    # generate expert mask ==> (num_tokens, k, num_experts)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # tokens routed to each expert ==> (k, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # apply penalty based on expert hidden sizes
    # normalize expert hidden sizes such that the p penalty loss
    # reduces to aux loss when all experts are the same size
    expert_hidden_sizes = torch.tensor(expert_hidden_sizes, device=device)
    normalized_dims = expert_hidden_sizes / expert_hidden_sizes.max()
    penalty = tokens_per_expert * normalized_dims

    # avg probability of routing to each experts ==> (num_experts)
    router_prob_per_expert = torch.mean(router_probs, dim=0)

    overall_loss = torch.sum(penalty * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def router_dynamic_loss(
    router_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the dynamic entropy loss for Top-P models.

    See the `Dynamic Routing paper`_ for more details.

    Parameters
    ----------
    router_probs : torch.Tensor
        Concatenated router probs for all sparse layers. Shape is [num_tokens, num_experts],
        where num_tokens is (batch_size * seq_len * num_sparse_layers)

    Returns
    -------
    torch.Tensor
        The dynamic entropy loss for the router.

    References
    ----------
    .. _Dynamic Routing paper:
        https://arxiv.org/abs/2403.07652
    """

    loss = -torch.sum(router_probs * torch.log(router_probs), dim=-1)
    print(loss)
    raise Exception()

    return loss.mean()
