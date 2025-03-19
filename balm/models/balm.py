# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import BalmConfig
from ..modules import (
    BalmLMHead, 
    BalmSequenceClassificationHead, 
    DenseTransformerLayer
)
from ..outputs import (
    BaseModelOutput, 
    MaskedLMOutput, 
    SequenceClassifierOutput
)
from .base import (
    BalmPreTrainedModel, 
    FreezeBaseModelMixin, 
    ParameterCountMixin
)

__all__ = [
    "BalmModel",
    "BalmForMaskedLM",
    "BalmForSequenceClassification",
]


class BalmModel(BalmPreTrainedModel, ParameterCountMixin):
    """
    Parameters:
    -----------
    config: BalmConfig
        Configuration object defining model architecture and hyperparameters.
    """

    def __init__(self, config: BalmConfig):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.position_embedding_type == "absolute"
            else None
        )

        # layers
        self.layers = nn.ModuleList([DenseTransformerLayer(config) for _ in range(config.num_hidden_layers)])

        # final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, tuple]:
        """
        Parameters:
        -----------

        input_ids: torch.LongTensor
            Tokenized input IDs, of shape (batch_size, sequence_length). Cannot be provided if
            `inputs_embeds` is also provided.

        attention_mask: torch.BoolTensor
            Attention mask, of shape (batch_size, sequence_length). If boolean, ``True`` indicates that
            tokens should be ignored for attention purposes. If float, it is added to the attention
            scores.

        position_ids: torch.LongTensor
            Position IDs, of shape (batch_size, sequence_length).

        inputs_embeds: torch.FloatTensor
            Input embeddings, of shape (batch_size, sequence_length, hidden_size). Cannot be provided
            if `input_ids` is also provided.

        output_attentions: bool
            Whether to output the attentions.

        output_hidden_states: bool
            Whether to output the hidden states.

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)

        Returns:
        --------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``BaseModelOutput`` object:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - hidden_states (torch.FloatTensor): hidden states
                - attentions (torch.FloatTensor): attention weights

            If `return_dict` is ``False``, the output is a ``tuple`` with the following elements:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - hidden_states (torch.FloatTensor): hidden states
                - attentions (torch.FloatTensor): attention weights

            For attentions and hidden_states, if they are not output, the corresponding
            value will be ``None`` (for ``BaseModelOutput``) or not returned at all (for ``tuple``).

        """
        # parse output options
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # inputs
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        if position_ids is None and self.position_embeddings is not None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # embeddings
        x = inputs_embeds
        if self.position_embeddings is not None and position_ids is not None:
            x = x + self.position_embeddings(position_ids)

        # layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (x,)

            x = layer(
                x,
                padding_mask=attention_mask,
                need_weights=output_attentions,
            )
            if output_attentions:
                x, attn = x
                all_self_attentions += (attn,)

        # final layer norm
        x = self.final_norm(x)

        # save the last hidden state
        if output_hidden_states:
            all_hidden_states += (x,)

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BalmForMaskedLM(BalmPreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin):
    """
    BALM model for masked language modeling.
    Uses the BALM encoder and adds a masked language modeling head.

    Parameters
    ----------
    config: BalmConfig
        Configuration object defining model architecture and hyperparameters.

    """

    config_class = BalmConfig
    base_model_prefix = "balm"

    def __init__(
        self,
        config: BalmConfig,
    ):
        super().__init__(config)
        self.config = config

        # model
        self.balm = BalmModel(config)
        self.lm_head = BalmLMHead(config)

        # loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[MaskedLMOutput, tuple]:
        """
        Forward pass

        Parameters
        ----------
        input_ids: torch.LongTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        position_ids: torch.LongTensor
            Position IDs, of shape (batch_size, sequence_length).

        inputs_embeds: torch.FloatTensor
            Input embeddings, of shape (batch_size, sequence_length, hidden_size). Cannot be provided
            if `input_ids` is also provided.

        output_attentions: bool
            Whether to output the attentions.

        output_hidden_states: bool
            Whether to output the hidden states.

        labels: torch.LongTensor
            Labels

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)

        Returns
        -------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``MaskedLMOutput`` object

        """
        # parse output options
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")

        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = outputs.last_hidden_state

        # lm head
        lm_logits = self.lm_head(x)

        # loss
        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss = self.criterion(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    lm_logits,
                    outputs.hidden_states,
                    outputs.attentions,
                ]
                if v is not None
            )
        return MaskedLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BalmForSequenceClassification(
    BalmPreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin
):
    """
    BALM model for sequence classification.
    Uses the BALM encoder and adds a sequence-level classification head.

    Parameters
    ----------
    config : BalmConfig
        Configuration object defining model architecture and hyperparameters.

    """

    config_class = BalmConfig
    base_model_prefix = "balm"

    def __init__(
        self,
        config: BalmConfig,
    ):
        super().__init__(config)
        self.config = config

        # model
        self.balm = BalmModel(config)
        self.classifier = BalmSequenceClassificationHead(config)
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[SequenceClassifierOutput, tuple]:
        """
        Forward pass

        Parameters
        ----------
        input_ids: torch.LongTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        key_padding_mask: torch.BoolTensor
            Key padding mask

        labels: torch.LongTensor
            Labels

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)
        """
        # parse output options
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")

        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = outputs.last_hidden_state

        # classifier
        classifier_logits = self.classifier(x)

        # classification loss
        classifier_loss = None
        if labels is not None:
            labels = labels.to(classifier_logits.device)
            classifier_loss = self.criterion(
                classifier_logits.view(-1, self.config.num_labels),
                labels.view(-1),
            )
        else:
            classifier_loss = None

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    classifier_logits,
                    outputs.hidden_states,
                    outputs.attentions,
                    classifier_loss,
                ]
                if v is not None
            )
        return SequenceClassifierOutput(
            loss=classifier_loss,
            logits=classifier_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            classifier_loss=classifier_loss,
        )
