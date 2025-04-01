# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

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
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the auxiliary load balancing loss.

    See the `Switch Transformer paper`_ for more details. This function
    implements the loss function presented in equations (4) - (6) of the paper.
    It aims at penalizing cases where the routing between experts is too unbalanced.

    Differing from the original `Switch Transformer paper`_, this is a more general
    implementation that supports both K=1 (top-1 routing) and K>1 (top-k routing):
      - for K=1 (top-1 routing), this reduces to the Switch Transformer load balancing loss.
      - for K>1 (top-k routing), this correctly accounts for multiple experts chosen per token.

    Parameters
    ----------
    router_probs : torch.Tensor
        Full probability distribution over experts for each token.
        Shape: [num_tokens, num_experts].

            # Probability assigned to each expert per token.
            # Shape: [batch_size, sequence_length, num_experts]

    expert_indices : torch.Tensor
        Indices of tokens selected by each expert.
        Shape: [num_experts, expert_capacity]

            # Indices of the selected experts for each token.
            # Shape: [batch_size, sequence_length, K] for top-k routing.
            # If K=1, shape can be [batch_size, sequence_length] or [batch_size, sequence_length, 1].

    Returns
    -------
    torch.Tensor
        The auxiliary load balancing loss for the router.

    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961
    """
    num_tokens, num_experts = router_probs.shape

    # create a mask of shape (num_experts, num_tokens)
    # where mask[i, j] = 1 if token j was selected for expert i, else 0
    expert_mask = torch.zeros(
        (num_experts, num_tokens), device=expert_indices.device, dtype=torch.float32
    )

    # filter out invalid (out of bounds) indices
    valid_mask = (expert_indices >= 0) & (expert_indices < num_tokens)
    token_indices = expert_indices[valid_mask]
    if len(token_indices) > 0:
        expert_mask.scatter_(1, token_indices.unsqueeze(0), 1.0)

    # compute G_j: average probability weight per expert from selected tokens
    G = expert_mask.sum(dim=1) / num_tokens  # ==> (num_experts,)

    # compute P_j: average router probability for expert j across all tokens
    P = router_probs.mean(dim=0)  # ==> (num_experts,)

    # compute the loss as described in the Switch Transformer paper:
    # L_aux = num_experts * sum_j(f_j * P_j)
    # where f_j is fraction of tokens assigned to expert j
    # and P_j is mean router probability for expert j
    loss = num_experts * torch.sum(G * P)
    return loss
