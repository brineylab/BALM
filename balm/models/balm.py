#!/usr/bin/python
# filename: balm.py

#
# Copyright (c) 2024 Bryan Briney
# License: GNU General Public License, version 3.0 (http://opensource.org/licenses/gpl-3-0/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


from typing import Optional

import torch
import torch.nn as nn

from ..config import BalmConfig
from ..modules import (
    BalmLMHead,
    BalmSequenceClassificationHead,
    DenseTransformerLayer,
)
from ..outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput
from .base import BalmBase

__all__ = [
    "BalmModel",
    "BalmForMaskedLM",
    "BalmForSequenceClassification",
]


class BalmModel(BalmBase):
    """
    Baseline Antibody Language Model (BALM)

    Parameters
        ----------
        config : BalmConfig
            The configuration object defining model architecture and hyperparameters.
    """

    config_class = BalmConfig

    def __init__(
        self,
        config: BalmConfig,
    ):
        super().__init__(config)
        # embedding
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.embed_dim,
            padding_idx=self.config.padding_idx,
        )

        # layers
        self.layers = nn.ModuleList(
            [
                DenseTransformerLayer(
                    self.config.embed_dim,
                    self.config.ffn_dim,
                    self.config.num_heads,
                    self.config.max_length,
                    dropout=self.config.dropout,
                    attention_dropout=self.config.attention_dropout,
                    token_embedding_dropout=self.config.token_embedding_dropout,
                    layer_norm_eps=self.config.layer_norm_eps,
                    activation=self.config.activation,
                    positional_embedding_type=self.config.positional_embedding_type,
                    pre_norm=self.config.pre_norm,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            self.config.embed_dim, eps=self.config.layer_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        # need_weights: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Expected shape is (batch_size, sequence_length).

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The output tensor. The shape is (batch_size, sequence_length, embed_dim).
            If `need_weights` is ``True``, the output is a tuple of the output tensor and the attention weights.
        """
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        x = self.embed_tokens(x)
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            x = layer(
                x,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                need_weights=output_attentions,
            )

            if output_attentions:
                x, attn = x
                all_self_attentions = all_self_attentions + (attn,)

        # final layer norm
        if self.config.pre_norm:
            x = self.final_layer_norm(x)

        # save the last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

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

        # # old output
        # if need_weights:
        #     return x, attn
        # return x


class BalmForMaskedLM(BalmBase):
    """
    BALM model for masked language modeling. Uses the base BALM model with rotary
    embeddings, pre-norm, and SwiGLU activations, and adds a language modeling head.

    Parameters
    ----------
    config : BalmConfig
        The configuration object defining model architecture and hyperparameters.
    """

    config_class = BalmConfig
    base_model_prefix = "balm"

    def __init__(
        self,
        config: BalmConfig,
    ):
        super().__init__(config)
        self.balm = BalmModel(config=self.config)
        self.lm_head = BalmLMHead(self.config.embed_dim, self.config.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # below are not used, only for compatibility with 🤗's transformers library
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> MaskedLMOutput:
        """
        Parameters
        ----------

        input_ids : torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len).

        attention_mask : Optional[torch.Tensor]
            The attention mask. Expected shape is (batch_size, seq_len, seq_len).

        key_padding_mask : Optional[torch.Tensor]
            The key padding mask. Expected shape is (batch_size, seq_len).

        labels : Optional[torch.Tensor]
            The labels. Expected shape is (batch_size).

        output_attentions : bool, default=False
            Whether to output the attentions.

        output_hidden_states : bool, default=False
            Whether to output the hidden states.

        return_dict : bool, default=True
            Whether to return a ``MaskedLMOutput`` object.

        Returns
        -------
        output (tuple or MaskedLMOutput):
            If `return_dict` is ``True``, the output is a ``MaskedLMOutput`` object, with the following properties:
                - loss (torch.FloatTensor): loss
                - logits (torch.FloatTensor): logits
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
            If `return_dict` is ``False``, the output is a ``tuple`` with the following elements (if they are not ``None``):
                - loss (torch.FloatTensor): loss
                - logits (torch.FloatTensor): logits
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
        """
        # # fix for 🤗's decision to use attention_mask where pytorch uses key_padding_mask
        # if key_padding_mask is None and attention_mask is not None:
        #     key_padding_mask = attention_mask
        #     attention_mask = None

        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # LM head
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)

        # masked LM loss
        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # this is from 🤗's RobertaForMaskedLM
            masked_lm_loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            # # old version, which should be equivalent to the above
            # masked_lm_loss = self.criterion(
            #     logits.view(-1, logits.size(-1)),
            #     labels.view(-1),
            # )

        # outputs
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # output = MaskedLMOutput(
        #     logits=logits,
        #     loss=masked_lm_loss,
        # )
        # if output_attentions:
        #     output.attentions = attn
        # if output_hidden_states:
        #     output.hidden_states = x
        # # tuple output
        # if not return_dict:
        #     outputs = logits
        #     if output_hidden_states:
        #         outputs += (x,)
        #     if output_attentions:
        #         outputs += (attn,)
        #     return (
        #         ((masked_lm_loss,) + outputs) if masked_lm_loss is not None else outputs
        #     )
        # # dict output
        # return output


class BalmForSequenceClassification(BalmBase):
    """
    BALM model for sequence classification. Uses the dense BALM transformer model and adds
    a sequence-level classification head.

    Parameters
    ----------
    config : BalmConfig
        The configuration object defining model architecture and hyperparameters.
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
        classifier_activation = (
            self.config.classifier_activation
            if self.config.classifier_activation is not None
            else "tanh"
        )
        # classifier_dropout = self.config.dropout
        # classifier_activation = "tanh"
        self.classifier = BalmSequenceClassificationHead(
            embed_dim=self.config.embed_dim,
            num_labels=self.config.num_labels,
            dropout=classifier_dropout,
            activation=classifier_activation,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # below are not used, only for compatibility with 🤗's transformers library
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        """
        Parameters
        ----------

        input_ids : torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len).

        attention_mask : Optional[torch.Tensor]
            The attention mask. Expected shape is (batch_size, seq_len, seq_len).

        key_padding_mask : Optional[torch.Tensor]
            The key padding mask. Expected shape is (batch_size, seq_len).

        labels : Optional[torch.Tensor]
            The labels. Expected shape is (batch_size).

        output_attentions : bool, default=False
            Whether to output the attentions.

        output_hidden_states : bool, default=False
            Whether to output the hidden states.

        return_dict : bool, default=True
            Whether to return a ``MaskedLMOutput`` object.

        Returns
        -------
        output (tuple or SequenceClassifierOutput):
            If `return_dict` is ``True``, the output is a ``SequenceClassifierOutput`` object, with the following properties:
                - loss (torch.FloatTensor): loss
                - logits (torch.FloatTensor): logits
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
            If `return_dict` is ``False``, the output is a ``tuple`` with the following elements (if they are not ``None``):
                - loss (torch.FloatTensor): loss
                - logits (torch.FloatTensor): logits
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
        """
        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # classifier
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        # classification loss
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.criterion(
                logits.view(-1, self.config.num_labels),
                labels.view(-1),
            )

        # output
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # output = ClassifierOutput(
        #     logits=logits,
        #     loss=classifier_loss,
        # )
        # if output_attentions:
        #     output.attentions = attn
        # if output_hidden_states:
        #     output.hidden_states = x
        # if return_dict:
        #     return output.as_dict()
        # return output.as_tuple()
