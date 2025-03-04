# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import BalmHybridMoEConfig
from ..loss import router_load_balancing_loss, router_z_loss
from ..modules import (
    BalmLMHead,
    BalmSequenceClassificationHead,
    HybridSparseTransformerLayer,
)
from ..outputs import MoEMaskedLMOutput, MoEModelOutput, MoESequenceClassifierOutput
from .base import BalmBase

__all__ = [
    "BalmHybridMoEModel",
    "BalmHybridMoEForMaskedLM",
    "BalmHybridMoEForSequenceClassification",
]


class BalmHybridMoEModel(BalmBase):
    """
    BALM Mixture of Experts model.
    """

    config_class = BalmHybridMoEConfig

    def __init__(
        self,
        config: BalmHybridMoEConfig,
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
                HybridSparseTransformerLayer(
                    embed_dim=self.config.embed_dim,
                    ffn_dim=self.config.ffn_dim,
                    residual_ffn_dim=self.config.residual_ffn_dim,
                    num_heads=self.config.num_heads,
                    num_experts=self.config.num_experts,
                    num_shared_experts=self.config.num_shared_experts,
                    max_length=self.config.max_length,
                    top_k=self.config.router_top_k,
                    expert_capacity=self.config.expert_capacity,
                    send_bos_to_all_experts=self.config.send_bos_to_all_experts,
                    activation=self.config.activation,
                    expert_activation=self.config.expert_activation,
                    dropout=self.config.dropout,
                    expert_ffn_dropout=self.config.expert_ffn_dropout,
                    attention_dropout=self.config.attention_dropout,
                    token_embedding_dropout=self.config.token_embedding_dropout,
                    layer_norm_eps=self.config.layer_norm_eps,
                    router_dtype=self.config.router_dtype,
                    router_bias=self.config.router_bias,
                    router_jitter=self.config.router_jitter,
                    router_ignore_padding_tokens=self.config.router_ignore_padding_tokens,
                    expert_choice_router=self.config.expert_choice_router,
                    pre_norm=self.config.pre_norm,
                    positional_embedding_type=self.config.positional_embedding_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # final layer norm
        self.final_norm = nn.LayerNorm(self.config.embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        output_expert_indexes: bool = False,
        return_dict: bool = True,
    ):
        """
        Parameters:
        -----------

        input_ids: torch.LomgTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        output_attentions: bool
            Whether to output attention weights

        output_hidden_states: bool
            Whether to output hidden states

        output_router_logits: bool
            Whether to output router logits

        output_expert_indexes: bool
            Whether to output expert indices

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple by default)


        Returns:
        --------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``MoEModelOutput`` object:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - z_loss (torch.FloatTensor): router z loss
                - aux_loss (torch.FloatTensor): router auxiliary loss
                - hidden_states (torch.FloatTensor): hidden states
                - attentions (torch.FloatTensor): attention weights
                - router_logits (torch.FloatTensor): router logits
                - expert_indexes (torch.LongTensor): expert indexes

            If `return_dict` is ``False``, the output is a ``tuple`` with the f0llowing elements:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - z_loss (torch.FloatTensor): router z loss
                - aux_loss (torch.FloatTensor): router auxiliary loss
                - hidden_states (torch.FloatTensor): hidden states
                - attentions (torch.FloatTensor): attention weights
                - router_logits (torch.FloatTensor): router logits
                - expert_indexes (torch.LongTensor): expert indexes

            For attentions, hidden_states, router_logits, and expert_indexes, if they are not output, the corresponding
            value will be ``None`` (for ``MoEModelOutput``) or not returned at all (for ``tuple``).

        """
        # init
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        router_logits = ()
        expert_indexes = ()

        # embeddings
        x = self.embed_tokens(input_ids)

        # encoder
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            x = layer(
                x,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                need_weights=output_attentions,
                # output_router_logits=output_router_logits,
            )
            if output_attentions:
                x, attn, router_tuple = x
                all_self_attentions = all_self_attentions + (attn,)
            else:
                x, router_tuple = x
            router_logits = router_logits + (router_tuple[0],)
            expert_indexes = expert_indexes + (router_tuple[1],)

        # final layer norm
        x = self.final_norm(x)

        # save the last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        # router losses
        cat_router_logits = torch.cat(router_logits, dim=1)
        cat_expert_indexes = torch.cat(expert_indexes, dim=1)
        router_probs = nn.Softmax(dim=-1)(cat_router_logits)
        z_loss = router_z_loss(cat_router_logits)
        if self.config.expert_choice_router:
            aux_loss = None
        else:
            aux_loss = router_load_balancing_loss(router_probs, cat_expert_indexes)

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    z_loss,
                    aux_loss,
                    all_hidden_states,
                    all_self_attentions,
                    router_logits if output_router_logits else None,
                    expert_indexes if output_expert_indexes else None,
                ]
                if v is not None
            )

        return MoEModelOutput(
            last_hidden_state=x,
            z_loss=z_loss,
            aux_loss=aux_loss,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=router_logits if output_router_logits else None,
            expert_indexes=expert_indexes if output_expert_indexes else None,
        )


class BalmHybridMoEForMaskedLM(BalmBase):
    """
    BALM Hybrid Mixture of Experts model for Masked Language Modeling.

    Parameters
    ----------
    config: BalmMoEConfig
        Configuration for the model.
    """

    config_class = BalmHybridMoEConfig
    base_model_prefix = "balm"

    def __init__(
        self,
        config: BalmHybridMoEConfig,
    ):
        super().__init__(config)
        self.balm = BalmHybridMoEModel(
            config=self.config,
        )
        self.lm_head = BalmLMHead(
            embed_dim=self.config.embed_dim,
            output_dim=self.config.vocab_size,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.z_loss_coef = self.config.router_z_loss_coef
        self.aux_loss_coef = self.config.router_aux_loss_coef

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = True,
        output_expert_indexes: bool = False,
        return_dict: bool = True,
    ) -> Union[MoEMaskedLMOutput, tuple]:
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
            If `return_dict` is ``True``, the output is a ``MoEMaskedLMOutput`` object
        """
        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
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
        lm_loss = None
        if labels is not None:
            # lm loss
            labels = labels.to(lm_logits.device)
            lm_loss = self.criterion(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )

            # router loss(es)
            z_loss = self.z_loss_coef * (outputs.z_loss)
            if self.config.expert_choice_router:
                loss = lm_loss + z_loss
            else:
                aux_loss = self.aux_loss_coef * (outputs.aux_loss)
                loss = lm_loss + z_loss + aux_loss
        else:
            loss = None
            z_loss = None
            aux_loss = None

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    z_loss,
                    aux_loss,
                    lm_loss,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits,
                    outputs.expert_indexes,
                ]
                if v is not None
            )

        return MoEMaskedLMOutput(
            loss=loss,
            z_loss=z_loss,
            aux_loss=aux_loss,
            lm_loss=lm_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_indexes=outputs.expert_indexes,
        )


class BalmHybridMoEForSequenceClassification(BalmBase):
    """
    BALM Hybrid Mixture-of-Experts (MoE) model for sequence classification.
    Uses the BALM Hybrid MoE encoder and adds a sequence-level classification head.

    Parameters
    ----------
    config : BalmHybridMoEConfig
        Configuration object defining model architecture and hyperparameters.

    """

    config_class = BalmHybridMoEConfig
    base_model_prefix = "balm"

    def __init__(
        self,
        config: BalmHybridMoEConfig,
    ):
        super().__init__(config)
        # model
        self.balm = BalmHybridMoEModel(config=self.config)

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
        self.classifier = BalmSequenceClassificationHead(
            embed_dim=self.config.embed_dim,
            num_labels=self.config.num_labels,
            dropout=classifier_dropout,
            activation=classifier_activation,
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # router loss coefficients
        self.z_loss_coef = self.config.router_z_loss_coef
        self.aux_loss_coef = self.config.router_aux_loss_coef

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        output_expert_indexes: bool = False,
        return_dict: bool = True,
    ) -> Union[MoESequenceClassifierOutput, tuple]:
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
        """

        # encoder
        outputs = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_expert_indexes=output_expert_indexes,
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

            # router loss(es)
            z_loss = self.z_loss_coef * (outputs.z_loss)
            if self.config.expert_choice_router:
                loss = classifier_loss + z_loss
                aux_loss = None
            else:
                aux_loss = self.aux_loss_coef * (outputs.aux_loss)
                loss = classifier_loss + z_loss + aux_loss
        else:
            loss = None
            z_loss = None
            aux_loss = None

        # outputs
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    z_loss,
                    aux_loss,
                    classifier_loss,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits,
                    outputs.expert_indexes,
                ]
                if v is not None
            )

        return MoESequenceClassifierOutput(
            loss=loss,
            z_loss=z_loss,
            aux_loss=aux_loss,
            classifier_loss=classifier_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_indexes=outputs.expert_indexes,
        )
