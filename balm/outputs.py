# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from transformers.utils.generic import ModelOutput

__all__ = [
    "MoEMaskedLMOutput", 
    "MoEModelOutput", 
    "MoESequenceClassifierOutput",
    "BaseModelOutput", 
    "MaskedLMOutput", 
    "SequenceClassifierOutput"
]


@dataclass
class BalmModelOutput(ModelOutput):
    """
    Base class for Balm model outputs, with potential hidden states and attentions.
    """

    def to(self, device: Union[torch.device, str]):
        for key, value in self.__dict__.items():
            # move tensors to device
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
            # move tuples of tensors to device
            elif isinstance(value, tuple) and all(
                isinstance(v, torch.Tensor) for v in value
            ):
                self.__dict__[key] = tuple(v.to(device) for v in value)
        # clear GPU cache if moving to CPU
        if torch.cuda.is_available() and torch.device(device).type == "cpu":
            torch.cuda.empty_cache()
        return self


@dataclass
class BaseModelOutput(BalmModelOutput):
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
class MaskedLMOutput(BalmModelOutput):
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
class SequenceClassifierOutput(BalmModelOutput):
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
    
    classifier_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The classifier attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).
    
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    classifier_attentions: Optional[torch.FloatTensor] = None


@dataclass
class MoEModelOutput(BalmModelOutput):
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
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None


@dataclass
class MoEMaskedLMOutput(BalmModelOutput):
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
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None
    lm_loss: torch.FloatTensor = None


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

    z_loss : torch.FloatTensor
        The z loss tensor. The shape is (1,).

    aux_loss : torch.FloatTensor
        The auxiliary loss tensor. The shape is (1,).

    classifier_loss : torch.FloatTensor
        The classifier loss tensor. The shape is (1,).
    
    classifier_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        The classifier attention weights tensor. The shape is (batch_size, num_heads, sequence_length, sequence_length).

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    expert_indexes: Optional[Tuple[torch.LongTensor]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    classifier_loss: torch.FloatTensor = None
    classifier_attentions: Optional[torch.FloatTensor] = None
