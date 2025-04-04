# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
from torch.nn import functional as F

__all__ = ["router_z_loss", "router_load_balancing_loss",  "router_load_balancing_loss_ignoredropped"]


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

# aux loss code (slightly modified) from Mixtral
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L851
# potential problem: Mixtral doesn't drop tokens (ie. no expert capacity), so this loss doesn't account for that
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

    Differing from the original `Switch Transformer paper`_, this is a more general
    implementation that supports both K=1 (top-1 routing) and K>1 (top-k routing):
      - for K=1 (top-1 routing), this reduces to the Switch Transformer load balancing loss.
      - for K>1 (top-k routing), this correctly accounts for multiple experts chosen per token.

    Parameters
    ----------
    router_logits : torch.Tensor
        Concatenated router logits for all sparse layers.
        Shape: [num_tokens, num_experts].
            # num_tokens = (batch_size * seq_len * num_sparse_layers)

    Returns
    -------
    torch.Tensor
        The auxiliary load balancing loss for the router.

    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961
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


# attempt at modifying mixtral aux loss to ignore dropped tokens
# this code requires the router to return expert_indices that track the dropped tokens
# initial results from trying this make me think we shouldn't do this
# it seems to make the Top2 model essentially act like a Top1 model
def router_load_balancing_loss_ignoredropped(
    router_probs: torch.Tensor, # (num_tokens, num_experts)
    expert_indices: torch.Tensor, # (num_tokens, k), dropped tokens 'route' to the value of num_experts (ie. a non-existent expert)
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
    router_logits : torch.Tensor
        Concatenated router logits for all sparse layers.
        Shape: [num_tokens, num_experts].
            # num_tokens = (batch_size * seq_len * num_sparse_layers)

    Returns
    -------
    torch.Tensor
        The auxiliary load balancing loss for the router.

    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961
    """
    num_tokens, num_experts = router_probs.shape

    # generate expert mask with extra 'dropped' expert ==> (num_tokens, k, num_experts + 1)
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts + 1)

    # remove 'dropped' expert ==> (num_tokens, k, num_experts)
    expert_mask = expert_mask[:, :, :-1]

    # Compute the percentage of tokens routed to each experts ==> (k, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts ==> (num_experts)
    router_prob_per_expert = torch.mean(router_probs, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts
    return overall_loss


# BROKEN
# calculates F (proportion of tokens recieved by each expert) incorrectly
# def router_load_balancing_loss(
#     router_probs: torch.Tensor,
#     expert_indices: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """
#     Computes the auxiliary load balancing loss.

#     See the `Switch Transformer paper`_ for more details. This function
#     implements the loss function presented in equations (4) - (6) of the paper.
#     It aims at penalizing cases where the routing between experts is too unbalanced.

#     Differing from the original `Switch Transformer paper`_, this is a more general
#     implementation that supports both K=1 (top-1 routing) and K>1 (top-k routing):
#       - for K=1 (top-1 routing), this reduces to the Switch Transformer load balancing loss.
#       - for K>1 (top-k routing), this correctly accounts for multiple experts chosen per token.

#     Parameters
#     ----------
#     router_probs : torch.Tensor
#         Full probability distribution over experts for each token.
#         Shape: [num_tokens, num_experts].

#             # Probability assigned to each expert per token.
#             # Shape: [batch_size, sequence_length, num_experts]

#     expert_indices : torch.Tensor
#         Indices of tokens selected by each expert.
#         Shape: [num_experts, expert_capacity]

#             # Indices of the selected experts for each token.
#             # Shape: [batch_size, sequence_length, K] for top-k routing.
#             # If K=1, shape can be [batch_size, sequence_length] or [batch_size, sequence_length, 1].

#     Returns
#     -------
#     torch.Tensor
#         The auxiliary load balancing loss for the router.

#     .. _Switch Transformer paper:
#         https://arxiv.org/abs/2101.03961
#     """
#     num_tokens, num_experts = router_probs.shape

#     # create a mask of shape (num_experts, num_tokens)
#     # where mask[i, j] = 1 if token j was selected for expert i, else 0
#     expert_mask = torch.zeros(
#         (num_experts, num_tokens), device=expert_indices.device, dtype=torch.float32
#     )

#     # filter out invalid (out of bounds) indices
#     valid_mask = (expert_indices >= 0) & (expert_indices < num_tokens)
#     token_indices = expert_indices[valid_mask]
#     if len(token_indices) > 0:
#         expert_mask.scatter_(1, token_indices.unsqueeze(0), 1.0)

#     # compute f_j: average probability weight per expert from selected tokens
#     F = expert_mask.sum(dim=1) / num_tokens  # ==> (num_experts,)

#     # compute P_j: average router probability for expert j across all tokens
#     P = router_probs.mean(dim=0)  # ==> (num_experts,)

#     print("f_j: ", F) ## this returns 0 for all experts except the first one!!
#     print("P_j: ", P)

#     # compute the loss as described in the Switch Transformer paper:
#     # L_aux = num_experts * sum_j(f_j * P_j)
#     # where f_j is fraction of tokens assigned to expert j
#     # and P_j is mean router probability for expert j
#     loss = num_experts * torch.sum(F * P)

#     print(loss)

#     raise Exception()
#     return loss
