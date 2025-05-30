# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import BalmMoEConfig
from ..loss import (
    router_load_balancing_loss,
    router_p_penalty_loss,
    router_z_loss,
    router_dynamic_loss,
)
from ..modules import (
    BalmLMHead,
    BalmSequenceClassificationHead,
    BalmAttentionSequenceClassificationHead,
    DenseTransformerLayer,
    SparseTransformerLayer,
)
from ..outputs import MoEModelOutput, MoEMaskedLMOutput, MoESequenceClassifierOutput
from .base import BalmPreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin

__all__ = [
    "BalmMoEModel",
    "BalmMoEForMaskedLM",
    "BalmMoEForSequenceClassification",
]


class BalmMoEModel(BalmPreTrainedModel, ParameterCountMixin):
    """
    Parameters:
    -----------
    config: BalmMoEConfig
        Configuration object defining model architecture and hyperparameters.
    """

    config_class = BalmMoEConfig
    base_model_prefix = "balm_moe"

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
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
        self.layers = nn.ModuleList()
        for layer_idx in range(self.config.num_hidden_layers):
            if self.config.num_initial_dense_layers > layer_idx:
                layer = DenseTransformerLayer(config)
            elif self.config.alternate_sparsity:  # alternate dense/sparse layers
                offset_idx = layer_idx - self.config.num_initial_dense_layers
                if offset_idx % 2 == 0:
                    layer = SparseTransformerLayer(config)
                else:
                    layer = DenseTransformerLayer(config)
            else:  # sparse layers
                layer = SparseTransformerLayer(config)
            self.layers.append(layer)

        # final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # initialize weights and apply final processing
        self.post_init()

    def _get_balancing_loss(self, cat_router_probs, attention_mask):

        # determine k (default to 1 for top-p)
        k = self.config.num_experts_per_tok if self.config.router_type == "top-k" else 1

        # empty output dict
        losses = {}

        # aux loss
        if self.config.homogeneous_experts or not self.config.router_use_penalty_loss:
            aux_loss = router_load_balancing_loss(
                router_probs=cat_router_probs,
                k=k,
                attention_mask=(
                    attention_mask if self.config.router_mask_aux_loss else None
                ),
            )
            if aux_loss is not None:
                losses["aux_loss"] = aux_loss

        # penalty loss
        if self.config.router_use_penalty_loss:
            penalty_loss = router_p_penalty_loss(
                router_probs=cat_router_probs,
                k=k,
                expert_hidden_sizes=self.config.expert_intermediate_size,
            )
            if penalty_loss is not None:
                losses["penalty_loss"] = penalty_loss

        return losses

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_indexes: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MoEMaskedLMOutput, tuple]:
        """
        Parameters:
        -----------
        input_ids: torch.LongTensor
            Tokenized input IDs
        attention_mask: torch.LongTensor
            Attention mask, of shape (batch_size, sequence_length). Values of `1` indicate valid tokens
            while values of `0` indicate padding that should be ignored for attention purposes.
        position_ids: torch.LongTensor
            Position IDs, of shape (batch_size, sequence_length).
        inputs_embeds: torch.FloatTensor
            Input embeddings, of shape (batch_size, sequence_length, hidden_size). Cannot be provided
            if `input_ids` is also provided.
        output_attentions: bool
            Whether to output attention weights
        output_hidden_states: bool
            Whether to output hidden states
        output_router_logits: bool
            Whether to output router logits
        output_expert_indexes: bool
            Whether to output expert indexes
        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)

        Returns:
        --------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``MoEModelOutput`` object.
            Otherwise, the output is a tuple.
        """

        # parse output options
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_expert_indexes = (
            output_expert_indexes
            if output_expert_indexes is not None
            else self.config.output_expert_indexes
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # init
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        router_logits = ()
        router_probs = ()
        expert_idxs = ()

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

            if isinstance(layer, SparseTransformerLayer):
                # sparse layer, so we need to collect router/expert info
                x = layer(
                    x,
                    padding_mask=attention_mask,
                    need_weights=output_attentions,
                )
                if output_attentions:
                    x, attn, router_tuple = x
                    all_self_attentions += (attn,)
                else:
                    x, router_tuple = x
                router_logits += (router_tuple[0],)
                router_probs += (router_tuple[1],)
                expert_idxs += (router_tuple[3],)
            else:
                # dense layer, no router info
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

        # concat logits & probs for router losses
        cat_router_logits = torch.cat(router_logits, dim=0)
        cat_router_probs = torch.cat(router_probs, dim=0)

        # router losses
        router_type = self.config.router_type
        moe_losses = {}
        if router_type == "expert-choice":
            moe_losses["z_loss"] = router_z_loss(cat_router_logits)
        elif router_type == "top-k":
            moe_losses.update(
                self._get_balancing_loss(
                    cat_router_probs=cat_router_probs,
                    attention_mask=attention_mask,
                )
            )
            if "aux_loss" in moe_losses:  # homogeneous experts
                moe_losses["z_loss"] = router_z_loss(cat_router_logits)
        else:  # top-p
            moe_losses.update(
                self._get_balancing_loss(
                    cat_router_probs=cat_router_probs,
                    attention_mask=attention_mask,
                )
            )
            moe_losses["dynamic_loss"] = router_dynamic_loss(cat_router_probs)

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    all_hidden_states,
                    all_self_attentions,
                    router_logits if output_router_logits else None,
                    expert_idxs if output_expert_indexes else None,
                    moe_losses if moe_losses else None,
                ]
                if v is not None
            )
        return MoEModelOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=router_logits if output_router_logits else None,
            expert_indexes=expert_idxs if output_expert_indexes else None,
            moe_losses=moe_losses if moe_losses else None,
        )


class BalmMoEForMaskedLM(
    BalmPreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin
):
    """
    BALM Mixture-of-Experts (MoE) model for masked language modeling.
    Uses the BALM-MoE encoder and adds a masked language modeling head.

    Parameters
    ----------
    config: BalmMoEConfig
        Configuration object defining model architecture and hyperparameters.
    """

    config_class = BalmMoEConfig
    base_model_prefix = "balm_moe"

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
        super().__init__(config)
        self.config = config

        # model
        self.balm_moe = BalmMoEModel(config)
        self.lm_head = BalmLMHead(config)

        # loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_indexes: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MoEMaskedLMOutput, tuple]:
        """
        Forward pass

        Parameters
        ----------
        input_ids: torch.LongTensor
            Tokenized input IDs
        attention_mask: torch.LongTensor
            Attention mask
        position_ids: torch.LongTensor
            Position IDs, of shape (batch_size, sequence_length).
        inputs_embeds: torch.FloatTensor
            Input embeddings, of shape (batch_size, sequence_length, hidden_size). Cannot be provided
            if `input_ids` is also provided.
        labels: torch.LongTensor
            Labels
        output_attentions: bool
            Whether to output attention weights
        output_hidden_states: bool
            Whether to output hidden states
        output_router_logits: bool
            Whether to output router logits
        output_expert_indexes: bool
            Whether to output expert indexes
        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)

        Returns
        -------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``MoEMaskedLMOutput`` object.
            Otherwise, the output is a tuple.
        """

        # parse output options
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_expert_indexes = (
            output_expert_indexes
            if output_expert_indexes is not None
            else self.config.output_expert_indexes
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # encoder
        outputs = self.balm_moe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_expert_indexes=output_expert_indexes,
            return_dict=True,
        )
        x = outputs.last_hidden_state

        # lm head
        lm_logits = self.lm_head(x)

        # loss
        loss = None
        moe_losses = outputs.moe_losses if outputs.moe_losses else {}

        if labels is not None:
            # lm loss
            labels = labels.to(lm_logits.device)
            moe_losses["lm_loss"] = self.criterion(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

            # apply coeffs
            for k, coef in self.router_loss_coeffs.items():
                if k in moe_losses:
                    moe_losses[k] = coef * moe_losses[k]

            # sum
            loss = sum(moe_losses.values())

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    lm_logits,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits,
                    outputs.expert_indexes,
                    moe_losses,
                ]
                if v is not None
            )
        return MoEMaskedLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_indexes=outputs.expert_indexes,
            moe_losses=moe_losses,
        )


class BalmMoEForSequenceClassification(
    BalmPreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin
):
    """
    BALM Mixture-of-Experts (MoE) model for sequence classification.
    Uses the BALM-MoE encoder and adds a sequence-level classification head.
    Can be configured with or without an attention block.

    Parameters
    ----------
    config : BalmMoEConfig
        Configuration object defining model architecture and hyperparameters.
    """

    config_class = BalmMoEConfig
    base_model_prefix = "balm_moe"

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
        super().__init__(config)
        self.config = config

        # model
        self.balm_moe = BalmMoEModel(config)
        if config.attention_classifier:
            self.classifier = BalmAttentionSequenceClassificationHead(config)
        else:
            if self.config.output_classifier_attentions == True:
                raise ValueError(
                    "Invalid classifier configuration. Cannot output classifier attentions when attention_classifier is False."
                )
            self.classifier = BalmSequenceClassificationHead(config)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # initialize weights
        self.init_weights()

        # freeze base model weights
        # and zero out router loss coeffs
        if config.classifier_freeze_base:
            self.freeze_base_model()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_classifier_attentions: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_indexes: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MoESequenceClassifierOutput, tuple]:
        """
        Forward pass

        Parameters
        ----------
        input_ids: torch.LongTensor
            Tokenized input IDs
        attention_mask: torch.LongTensor
            Attention mask
        position_ids: torch.LongTensor
            Position IDs, of shape (batch_size, sequence_length).
        inputs_embeds: torch.FloatTensor
            Input embeddings, of shape (batch_size, sequence_length, hidden_size). Cannot be provided
            if `input_ids` is also provided.
        labels: torch.LongTensor
            Labels
        output_classifier_attentions: bool
            Whether to output classifier attention weights.
        output_attentions: bool
            Whether to output attention weights
        output_hidden_states: bool
            Whether to output hidden states
        output_router_logits: bool
            Whether to output router logits
        output_expert_indexes: bool
            Whether to output expert indexes
        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple if False)

        Returns:
        --------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``MoESequenceClassifierOutput`` object.
            Otherwise, the output is a tuple.
        """

        # parse output options
        output_classifier_attentions = (
            output_classifier_attentions
            if output_classifier_attentions is not None
            else self.config.output_classifier_attentions
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_expert_indexes = (
            output_expert_indexes
            if output_expert_indexes is not None
            else self.config.output_expert_indexes
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # encoder
        outputs = self.balm_moe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_expert_indexes=output_expert_indexes,
            return_dict=True,
        )
        x = outputs.last_hidden_state

        # invert attention mask to padding mask and convert to boolean
        if attention_mask is not None:
            padding_mask = 1 - attention_mask
            padding_mask = padding_mask.bool()

        # classifier
        classifier_out = self.classifier(
            x, padding_mask=padding_mask, need_weights=output_classifier_attentions
        )
        if output_classifier_attentions:
            classifier_logits, classifier_attn = classifier_out
        else:
            classifier_logits = classifier_out
            classifier_attn = None

        # loss
        loss = None
        moe_losses = outputs.moe_losses if outputs.moe_losses else {}
        if labels is not None:
            labels = labels.to(classifier_logits.device)
            moe_losses["classifier_loss"] = self.criterion(
                classifier_logits.view(-1, self.config.num_labels),
                labels.view(-1),
            )

            # apply coeffs
            for k, coef in self.router_loss_coeffs.items():
                if k in moe_losses:
                    moe_losses[k] = coef * moe_losses[k]

            # sum
            loss = sum(moe_losses.values())

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    classifier_logits,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits,
                    outputs.expert_indexes,
                    classifier_attn,
                    moe_losses,
                ]
                if v is not None
            )
        return MoESequenceClassifierOutput(
            loss=loss,
            logits=classifier_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_indexes=outputs.expert_indexes,
            classifier_attentions=classifier_attn,
            moe_losses=moe_losses,
        )
