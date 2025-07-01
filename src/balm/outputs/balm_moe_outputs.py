# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .base_outputs import BalmModelOutput

__all__ = ["MoEMaskedLMOutput", "MoEModelOutput", "MoESequenceClassifierOutput"]


@dataclass
class MoEModelOutput(BalmModelOutput):
    """
    Base class for MoE model outputs, with potential hidden states and attentions.

    Parameters
    ----------
    last_hidden_state : torch.FloatTensor
        The last hidden state tensor. The shape is (batch_size, sequence_length, hidden_size).
    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, hidden_size).
    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).
    router_logits : Optional[Tuple[torch.FloatTensor]]
        The router logits tensor. The shape is (batch_size, sequence_length, num_experts).
    expert_indexes : Optional[Tuple[torch.LongTensor]]
        The expert indexes tensor. The shape is (batch_size, sequence_length, num_experts).
    moe_losses : Optional[Dict[torch.FloatTensor]]
        A dict containing the MoE losses. Can include any of the following losses:
        aux_loss, penalty_loss, z_loss, dynamic_loss
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None
    moe_losses: Optional[Dict[str, torch.FloatTensor]] = None


@dataclass
class MoEMaskedLMOutput(BalmModelOutput):
    """
    Base class for MoE model outputs, with potential hidden states and attentions.

    Parameters
    ----------
    loss : torch.FloatTensor
        The loss tensor. The shape is (1,).
    logits : torch.FloatTensor
        The output tensor. The shape is (batch_size, sequence_length, hidden_size).
    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).
    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, hidden_size).
    router_logits : Optional[Tuple[torch.FloatTensor]]
        The router logits tensor. The shape is (batch_size, sequence_length, num_experts).
    expert_indexes : Optional[Tuple[torch.LongTensor]]
        The expert indexes tensor. The shape is (batch_size, sequence_length, num_experts).
    moe_losses : Optional[Dict[torch.FloatTensor]]
        A dict containing the MoE losses. Can include any of the following losses:
        lm_loss, aux_loss, penalty_loss, z_loss, dynamic_loss
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None
    moe_losses: Optional[Dict[str, torch.FloatTensor]] = None


@dataclass
class MoESequenceClassifierOutput(BalmModelOutput):
    """
    Base class for MoE model outputs, with potential hidden states and attentions.

    Parameters
    ----------
    loss : torch.FloatTensor
        The loss tensor. The shape is (1,).
    logits : torch.FloatTensor
        The output tensor. The shape is (batch_size, num_labels).
    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, hidden_size).
    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).
    router_logits : Optional[Tuple[torch.FloatTensor]]
        The router logits tensor. The shape is (batch_size, sequence_length, num_experts).
    expert_indexes : Optional[Tuple[torch.LongTensor]]
        The expert indexes tensor. The shape is (batch_size, sequence_length, num_experts).
    classifier_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The classifier attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).
    moe_losses : Optional[Dict[torch.FloatTensor]]
        A dict containing the MoE losses. Can include any of the following losses:
        classifier_loss, aux_loss, penalty_loss, z_loss, dynamic_loss
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None
    classifier_attentions: Optional[torch.FloatTensor] = None
    moe_losses: Optional[Dict[str, torch.FloatTensor]] = None
