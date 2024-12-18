# Copyright (c) 2024 brineylab @ scripps
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
        Input logits of shape [batch_size, sequence_length, num_experts]

    Returns
    -------
    torch.Tensor
        The z-loss for the router


    .. _Designing Effective Sparse Expert Models:
        https://arxiv.org/abs/2202.08906
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def router_load_balancing_loss(
    router_probs: torch.Tensor, expert_indices: torch.Tensor
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
        Probability assigned to each expert per token.
        Shape: [batch_size, sequence_length, num_experts].

    expert_indices : torch.Tensor
        Indices of the selected experts for each token.
        Shape: [batch_size, sequence_length, K] for top-k routing.
        If K=1, shape can be [batch_size, sequence_length] or [batch_size, sequence_length, 1].

    Returns
    -------
    torch.Tensor
        The auxiliary load balancing loss for the router.

    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961
    """
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)
    if expert_indices.ndim == 2:
        expert_indices = expert_indices.unsqueeze(-1)  # [B, S] -> [B, S, 1]

    # create a one-hot mask for the chosen experts
    num_experts = router_probs.shape[-1]
    expert_mask = F.one_hot(expert_indices, num_classes=num_experts)  # [B, S, K, E]

    # sum over the K dimension to handle top-k > 1.
    # expert_mask goes from [B, S, K, E] -> [B, S, E], where values can be:
    #   - 1 if that expert is chosen among the top-k experts
    #   - 0 otherwise.
    # for K>1, if multiple experts are chosen, there will be multiple ones.
    expert_mask = expert_mask.sum(dim=-2).to(router_probs.dtype)  # [B, S, E]

    # actual (G_j) and expected (P_j) distribution of tokens per expert
    tokens_per_group_and_expert = expert_mask.mean(dim=[0, 1])  # [E]
    router_prob_per_group_and_expert = router_probs.mean(dim=[0, 1])  # [E]

    # according to the Switch Transformer paper, the load balancing loss is:
    #     L_aux = mean(G_j * P_j) * (num_experts^2)
    # this encourages the observed fraction (G_j) to match the predicted fraction (P_j)
    return torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert
    ) * (num_experts**2)


# def router_load_balancing_loss(
#     router_probs: torch.Tensor, expert_indices: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Computes the auxiliary load balancing loss.

#     See the `Switch Transformer manuscript`_ for more details. This function
#     implements the loss function presented in equations (4) - (6) of the paper.
#     It aims at penalizing cases where the routing between experts is too unbalanced.


#     Parameters:
#     -----------
#     router_probs : torch.Tensor
#         Probability assigned to each expert per token.
#         Shape: [batch_size, seqeunce_length, num_experts].

#     expert_indices : torch.Tensor
#         Indices tensor of identifying the selected expert for a given token.
#         Shape: [batch_size, seqeunce_length]

#     Returns
#     -------
#     torch.Tensor
#         The auxiliary load balancing loss for the router


#     .. _Switch Transformer manuscript:
#         https://arxiv.org/abs/2101.03961
#     """
#     num_experts = router_probs.shape[-1]
#     if expert_indices.dtype != torch.int64:  # F.one_hot fails if not int64
#         expert_indices = expert_indices.to(torch.int64)
#     if len(expert_indices.shape) == 2:
#         expert_indices = expert_indices.unsqueeze(2)
#     # expert mask
#     expert_mask = F.one_hot(expert_indices, num_experts)
#     expert_mask = torch.max(expert_mask, axis=-2).values
#     expert_mask = expert_mask.to(torch.float32)  # torch.mean needs float32
#     # compute aux loss
#     tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
#     router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
#     return torch.mean(
#         tokens_per_group_and_expert * router_prob_per_group_and_expert
#     ) * (num_experts**2)


# import torch
# from torch import nn

# __all__ = ["router_z_loss", "router_load_balancing_loss"]


# def router_z_loss(router_logits: torch.Tensor) -> float:
#     """
#     Computes the router z-loss.

#     The router z-loss was introduced in `Designing Effective Sparse Expert Models`_.
#     It encourages router logits to remain small in an effort to improve stability.


#     Parameters:
#     -----------
#     router_logits : float
#         Input logits of shape [batch_size, sequence_length, num_experts]

#     Returns:
#     --------
#         Scalar router z-loss


#     .. _Designing Effective Sparse Expert Models:
#         https://arxiv.org/abs/2202.08906
#     """
#     num_groups, tokens_per_group, _ = router_logits.shape
#     log_z = torch.logsumexp(router_logits, dim=-1)
#     z_loss = log_z**2
#     return torch.sum(z_loss) / (num_groups * tokens_per_group)


# def router_load_balancing_loss(
#     router_probs: torch.Tensor, expert_indices: torch.Tensor
# ) -> float:
#     """
#     Computes the auxiliary load balancing loss.

#     See the `Switch Transformer manuscript`_ for more details. This function
#     implements the loss function presented in equations (4) - (6) of the paper.
#     It aims at penalizing cases where the routing between experts is too unbalanced.


#     Parameters:
#     -----------
#     router_probs : torch.Tensor
#         Probability assigned to each expert per token.
#         Shape: [batch_size, seqeunce_length, num_experts].

#     expert_indices : torch.Tensor
#         Indices tensor of identifying the selected expert for a given token.
#         Shape: [batch_size, seqeunce_length]

#     Returns:
#         The auxiliary loss.


#     .. _Switch Transformer manuscript:
#         https://arxiv.org/abs/2101.03961
#     """
#     num_experts = router_probs.shape[-1]

#     # cast the expert indices to int64, otherwise one-hot encoding will fail
#     if expert_indices.dtype != torch.int64:
#         expert_indices = expert_indices.to(torch.int64)

#     if len(expert_indices.shape) == 2:
#         expert_indices = expert_indices.unsqueeze(2)

#     expert_mask = nn.functional.one_hot(expert_indices, num_experts)

#     # For a given token, determine if it was routed to a given expert.
#     expert_mask = torch.max(expert_mask, axis=-2).values

#     # cast to float32 otherwise mean will fail
#     expert_mask = expert_mask.to(torch.float32)
#     tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

#     router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
#     return torch.mean(
#         tokens_per_group_and_expert * router_prob_per_group_and_expert
#     ) * (num_experts**2)
