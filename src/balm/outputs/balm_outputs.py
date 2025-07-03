# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .base_outputs import BalmModelOutput

__all__ = ["BaseModelOutput", "MaskedLMOutput", "SequenceClassifierOutput"]


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
