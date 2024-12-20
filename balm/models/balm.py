# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from ..config import BalmConfig
from ..modules import BalmLMHead, BalmSequenceClassificationHead, DenseTransformerLayer
from ..outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput
from .base import FreezeBaseModelMixin, ParameterCountMixin

__all__ = [
    "BalmModel",
    "BalmForMaskedLM",
    "BalmForSequenceClassification",
]


class BalmModel(PreTrainedModel, ParameterCountMixin):
    config_class = BalmConfig
    base_model_prefix = None

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

        # embedding
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.position_embedding_type == "absolute"
            else None
        )

        # dropout
        self.dropout = nn.Dropout(config.dropout)
        self.embed_dropout = nn.Dropout(config.token_dropout)

        # layers
        self.layers = nn.ModuleList(
            [
                DenseTransformerLayer(
                    config=config,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        # final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # init weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, tuple]:
        """
        Parameters:
        -----------

        input_ids: torch.LomgTensor
            Tokenized input IDs, of shape (batch_size, sequence_length). Cannot be provided if
            `inputs_embeds` is also provided.

        attention_mask: torch.BoolTensor
            Attention mask, of shape (batch_size, sequence_length). If boolean, ``True`` indicates that
            tokens should be ignored for attention purposes. If float, it is added to the attention
            scores.

        token_type_ids: torch.LongTensor
            Token type IDs, of shape (batch_size, sequence_length).

        position_ids: torch.LongTensor
            Position IDs, of shape (batch_size, sequence_length).

        inputs_embeds: torch.FloatTensor
            Input embeddings, of shape (batch_size, sequence_length, hidden_size). Cannot be provided
            if `input_ids` is also provided.

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
        # init
        all_self_attentions = () if self.config.output_attentions else None
        all_hidden_states = () if self.config.output_hidden_states else None
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # inputs
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if position_ids is None and self.position_embeddings is not None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embeddings
        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
        if self.position_embeddings is not None and position_ids is not None:
            embeddings = embeddings + self.position_embeddings(position_ids)
        x = self.embed_dropout(embeddings)

        # layers
        for layer in self.layers:
            if self.config.output_hidden_states:
                all_hidden_states += (x,)

            x = layer(
                x,
                attention_mask=attention_mask,
            )
            if self.config.output_attentions:
                x, attn = x
                all_self_attentions += (attn,)

        # final layer norm
        if self.config.pre_norm:
            x = self.final_norm(x)

        # save the last hidden state
        if self.config.output_hidden_states:
            all_hidden_states += (x,)

        # outputs
        if not self.config.return_dict:
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


# class BalmModel(BalmBase):
#     """
#     Baseline Antibody Language Model (BALM)

#     Parameters
#     ----------
#     config : BalmConfig
#             The configuration object defining model architecture and hyperparameters.

#     """

#     config_class = BalmConfig

#     def __init__(
#         self,
#         config: BalmConfig,
#     ):
#         super().__init__(config)
#         # embedding
#         self.embed_tokens = nn.Embedding(
#             self.config.vocab_size,
#             self.config.embed_dim,
#             padding_idx=self.config.padding_idx,
#         )

#         # layers
#         self.layers = nn.ModuleList(
#             [
#                 DenseTransformerLayer(
#                     self.config.embed_dim,
#                     self.config.ffn_dim,
#                     self.config.num_heads,
#                     self.config.max_length,
#                     dropout=self.config.dropout,
#                     attention_dropout=self.config.attention_dropout,
#                     token_embedding_dropout=self.config.token_embedding_dropout,
#                     layer_norm_eps=self.config.layer_norm_eps,
#                     activation=self.config.activation,
#                     positional_embedding_type=self.config.positional_embedding_type,
#                     pre_norm=self.config.pre_norm,
#                 )
#                 for _ in range(self.config.num_layers)
#             ]
#         )

#         self.final_layer_norm = nn.LayerNorm(
#             self.config.embed_dim, eps=self.config.layer_norm_eps
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#     ) -> BaseModelOutput:
#         """
#         Parameters
#         ----------
#         x : torch.Tensor
#             The input tensor. Expected shape is (batch_size, sequence_length).

#         Returns
#         -------
#         Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
#             The output tensor. The shape is (batch_size, sequence_length, embed_dim).
#             If `need_weights` is ``True``, the output is a tuple of the output tensor and the attention weights.
#         """
#         all_self_attentions = () if output_attentions else None
#         all_hidden_states = () if output_hidden_states else None

#         x = self.embed_tokens(x)
#         for layer in self.layers:
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (x,)

#             x = layer(
#                 x,
#                 attention_mask=attention_mask,
#                 key_padding_mask=key_padding_mask,
#                 need_weights=output_attentions,
#             )

#             if output_attentions:
#                 x, attn = x
#                 all_self_attentions = all_self_attentions + (attn,)

#         # final layer norm
#         if self.config.pre_norm:
#             x = self.final_layer_norm(x)

#         # save the last hidden state
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (x,)

#         # outputs
#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     x,
#                     all_hidden_states,
#                     all_self_attentions,
#                 ]
#                 if v is not None
#             )
#         return BaseModelOutput(
#             last_hidden_state=x,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )


class BalmForMaskedLM(PreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin):
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
        # model
        self.balm = BalmModel(config=self.config)

        # LM head
        self.lm_head = BalmLMHead(
            hidden_size=self.config.hidden_size,
            output_dim=self.config.vocab_size,
            activation=self.config.classifier_activation,
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, tuple]:
        """
        Forward pass

        Parameters
        ----------
        input_ids: torch.LongTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        key_padding_mask: torch.BoolTensor
            Key padding mask. Not used (use attention_mask instead)

        labels: torch.LongTensor
            Labels

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)

        Returns
        -------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``MaskedLMOutput`` object

        """
        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
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


# class BalmForMaskedLM(BalmBase):
#     """
#     BALM model for masked language modeling. Uses the base BALM model with rotary
#     embeddings, pre-norm, and SwiGLU activations, and adds a language modeling head.

#     Parameters
#     ----------
#     config : BalmConfig
#         The configuration object defining model architecture and hyperparameters.
#     """

#     config_class = BalmConfig
#     base_model_prefix = "balm"

#     def __init__(
#         self,
#         config: BalmConfig,
#     ):
#         super().__init__(config)
#         self.balm = BalmModel(config=self.config)
#         self.lm_head = BalmLMHead(self.config.embed_dim, self.config.vocab_size)
#         self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#         # below are not used, only for compatibility with 🤗's transformers library
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.Tensor] = None,
#     ) -> MaskedLMOutput:
#         """
#         Parameters
#         ----------

#         input_ids : torch.Tensor
#             The input tensor. Expected shape is (batch_size, seq_len).

#         attention_mask : Optional[torch.Tensor]
#             The attention mask. Expected shape is (batch_size, seq_len, seq_len).

#         key_padding_mask : Optional[torch.Tensor]
#             The key padding mask. Expected shape is (batch_size, seq_len).

#         labels : Optional[torch.Tensor]
#             The labels. Expected shape is (batch_size).

#         output_attentions : bool, default=False
#             Whether to output the attentions.

#         output_hidden_states : bool, default=False
#             Whether to output the hidden states.

#         return_dict : bool, default=True
#             Whether to return a ``MaskedLMOutput`` object.

#         Returns
#         -------
#         output (tuple or MaskedLMOutput):
#             If `return_dict` is ``True``, the output is a ``MaskedLMOutput`` object, with the following properties:
#                 - loss (torch.FloatTensor): loss
#                 - logits (torch.FloatTensor): logits
#                 - attentions (torch.FloatTensor): attention weights
#                 - hidden_states (torch.FloatTensor): hidden states
#             If `return_dict` is ``False``, the output is a ``tuple`` with the following elements (if they are not ``None``):
#                 - loss (torch.FloatTensor): loss
#                 - logits (torch.FloatTensor): logits
#                 - attentions (torch.FloatTensor): attention weights
#                 - hidden_states (torch.FloatTensor): hidden states
#         """
#         # # fix for 🤗's decision to use attention_mask where pytorch uses key_padding_mask
#         # if key_padding_mask is None and attention_mask is not None:
#         #     key_padding_mask = attention_mask
#         #     attention_mask = None

#         # encoder
#         outputs = self.balm(
#             input_ids,
#             attention_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         # LM head
#         sequence_output = outputs[0]
#         logits = self.lm_head(sequence_output)

#         # masked LM loss
#         masked_lm_loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             # this is from 🤗's RobertaForMaskedLM
#             masked_lm_loss = self.criterion(
#                 logits.view(-1, self.config.vocab_size),
#                 labels.view(-1),
#             )

#         # outputs
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (
#                 ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
#             )

#         return MaskedLMOutput(
#             loss=masked_lm_loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class BalmForSequenceClassification(
    PreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin
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
        # model
        self.balm = BalmModel(config=self.config)

        # classifier
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.dropout
        )
        self.classifier = BalmSequenceClassificationHead(
            hidden_size=self.config.hidden_size,
            num_labels=self.config.num_labels,
            dropout=classifier_dropout,
            activation=self.config.classifier_activation,
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
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

        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            return_dict=True,
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
            loss = None

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
            loss=loss,
            logits=classifier_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            classifier_loss=classifier_loss,
        )


# class BalmForSequenceClassification(BalmBase):
#     """
#     BALM model for sequence classification. Uses the dense BALM transformer model and adds
#     a sequence-level classification head.

#     Parameters
#     ----------
#     config : BalmConfig
#         The configuration object defining model architecture and hyperparameters.
#     """

#     config_class = BalmConfig
#     base_model_prefix = "balm"

#     def __init__(
#         self,
#         config: BalmConfig,
#     ):
#         super().__init__(config)
#         # model
#         self.balm = BalmModel(config=self.config)

#         # classifier
#         classifier_dropout = (
#             self.config.classifier_dropout
#             if self.config.classifier_dropout is not None
#             else self.config.dropout
#         )
#         classifier_activation = (
#             self.config.classifier_activation
#             if self.config.classifier_activation is not None
#             else "tanh"
#         )
#         self.classifier = BalmSequenceClassificationHead(
#             embed_dim=self.config.embed_dim,
#             num_labels=self.config.num_labels,
#             dropout=classifier_dropout,
#             activation=classifier_activation,
#         )

#         # loss function
#         self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#         # below are not used, only for compatibility with 🤗's transformers library
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.Tensor] = None,
#     ) -> SequenceClassifierOutput:
#         """
#         Parameters
#         ----------

#         input_ids : torch.Tensor
#             The input tensor. Expected shape is (batch_size, seq_len).

#         attention_mask : Optional[torch.Tensor]
#             The attention mask. Expected shape is (batch_size, seq_len, seq_len).

#         key_padding_mask : Optional[torch.Tensor]
#             The key padding mask. Expected shape is (batch_size, seq_len).

#         labels : Optional[torch.Tensor]
#             The labels. Expected shape is (batch_size).

#         output_attentions : bool, default=False
#             Whether to output the attentions.

#         output_hidden_states : bool, default=False
#             Whether to output the hidden states.

#         return_dict : bool, default=True
#             Whether to return a ``MaskedLMOutput`` object.

#         Returns
#         -------
#         output (tuple or SequenceClassifierOutput):
#             If `return_dict` is ``True``, the output is a ``SequenceClassifierOutput`` object, with the following properties:
#                 - loss (torch.FloatTensor): loss
#                 - logits (torch.FloatTensor): logits
#                 - attentions (torch.FloatTensor): attention weights
#                 - hidden_states (torch.FloatTensor): hidden states
#             If `return_dict` is ``False``, the output is a ``tuple`` with the following elements (if they are not ``None``):
#                 - loss (torch.FloatTensor): loss
#                 - logits (torch.FloatTensor): logits
#                 - attentions (torch.FloatTensor): attention weights
#                 - hidden_states (torch.FloatTensor): hidden states
#         """
#         # encoder
#         outputs = self.balm(
#             input_ids,
#             attention_mask=attention_mask,
#             key_padding_mask=key_padding_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         # classifier
#         sequence_output = outputs[0]
#         logits = self.classifier(sequence_output)

#         # classification loss
#         classification_loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             classification_loss = self.criterion(
#                 logits.view(-1, self.config.num_labels),
#                 labels.view(-1),
#             )

#         # output
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (
#                 ((classification_loss,) + output)
#                 if classification_loss is not None
#                 else output
#             )

#         return SequenceClassifierOutput(
#             loss=classification_loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
