# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils.generic import ModelOutput

__all__ = ["BaseModelOutput", "MaskedLMOutput", "SequenceClassifierOutput"]


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Parameters
    ----------
    last_hidden_state : torch.FloatTensor
        The last hidden state tensor. The shape is (batch_size, sequence_length, hidden_size).

    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, hidden_size).

    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).

    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for outputs of masked language models.

    Parameters
    ----------
    logits : torch.Tensor
        The output tensor. The shape is (batch_size, sequence_length, vocab_size).

    loss : torch.Tensor
        The loss tensor. The shape is (1,).

    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, embed_dim).

    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence classification models.

    Parameters
    ----------
    loss : torch.Tensor
        The loss tensor. The shape is (1,).

    logits : torch.Tensor
        The output tensor. The shape is (batch_size, config.num_labels).

    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, hidden_size).

    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MoEModelOutput(ModelOutput):
    """
    Base class for MoE model outputs, with potential hidden states and attentions.

    Parameters
    ----------
    last_hidden_state : torch.FloatTensor
        The last hidden state tensor. The shape is (batch_size, sequence_length, hidden_size).

    z_loss : torch.FloatTensor
        The z loss tensor. The shape is (1,).

    aux_loss : torch.FloatTensor
        The auxiliary loss tensor. The shape is (1,).

    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, hidden_size).

    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).

    router_logits : Optional[Tuple[torch.FloatTensor]]
        The router logits tensor. The shape is (batch_size, sequence_length, num_experts).

    expert_indexes : Optional[Tuple[torch.LongTensor]]
        The expert indexes tensor. The shape is (batch_size, sequence_length, num_experts).

    """

    last_hidden_state: torch.FloatTensor = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None


@dataclass
class MoEMaskedLMModelOutput(ModelOutput):
    """
    Base class for MoE model outputs, with potential hidden states and attentions.

    Parameters
    ----------
    loss : torch.FloatTensor
        The loss tensor. The shape is (1,).

    z_loss : torch.FloatTensor
        The z loss tensor. The shape is (1,).

    aux_loss : torch.FloatTensor
        The auxiliary loss tensor. The shape is (1,).

    lm_loss : torch.FloatTensor
        The masked language model loss tensor. The shape is (1,).

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

    """

    loss: Optional[torch.FloatTensor] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    lm_loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None


@dataclass
class MoESequenceClassifierModelOutput(ModelOutput):
    """
    Base class for MoE model outputs, with potential hidden states and attentions.

    Parameters
    ----------
    loss : torch.FloatTensor
        The loss tensor. The shape is (1,).

    logits : torch.FloatTensor
        The output tensor. The shape is (batch_size, num_labels).

    z_loss : torch.FloatTensor
        The z loss tensor. The shape is (1,).

    aux_loss : torch.FloatTensor
        The auxiliary loss tensor. The shape is (1,).

    classifier_loss : torch.FloatTensor
        The classifier loss tensor. The shape is (1,).

    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        The hidden states tensor. The shape is (batch_size, sequence_length, hidden_size).

    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).

    router_logits : Optional[Tuple[torch.FloatTensor]]
        The router logits tensor. The shape is (batch_size, sequence_length, num_experts).

    expert_indexes : Optional[Tuple[torch.LongTensor]]
        The expert indexes tensor. The shape is (batch_size, sequence_length, num_experts).

    """

    loss: Optional[torch.FloatTensor] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    classifier_loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None
