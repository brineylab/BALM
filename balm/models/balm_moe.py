# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from ..activation import get_activation_fn
from ..config import BalmMoEConfig
from ..loss import router_load_balancing_loss, router_z_loss
from ..modules import (
    BalmLMHead,
    BalmSequenceClassificationHead,
    DenseTransformerLayer,
    SparseTransformerLayer,
)
from ..outputs import MoEMaskedLMOutput, MoEModelOutput, MoESequenceClassifierOutput
from .base import BalmBase, FreezeBaseModelMixin, ParameterCountMixin

__all__ = [
    "BalmMoEModel",
    "BalmMoEForMaskedLM",
    "BalmMoEForSequenceClassification",
]


class BalmMoEModel(PreTrainedModel, ParameterCountMixin):
    config_class = BalmMoEConfig
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
        if config.alternate_sparsity:
            # alternate dense/sparse layers (dense first)
            layers = []
            for layer_num in range(config.num_hidden_layers):
                if layer_num % 2 == 0:
                    layers.append(
                        DenseTransformerLayer(
                            config=config,
                        )
                    )
                else:
                    layers.append(
                        SparseTransformerLayer(
                            config=config,
                        )
                    )
            self.layers = nn.ModuleList(layers)
        else:
            # all sparse layers
            self.layers = nn.ModuleList(
                [
                    SparseTransformerLayer(
                        config=config,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )

        # final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # init weights
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    # def freeze_base_model(self):
    #     # Freeze all base model parameters
    #     for param in self.parameters():
    #         param.requires_grad = False
    #     # Also zero out aux and z-loss in all routers
    #     self.router_z_loss_coef = 0.0
    #     self.router_aux_loss_coef = 0.0

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # output_attentions: bool = False,
        # output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
        # key_padding_mask: Optional[torch.Tensor] = None,
        # output_attentions: bool = False,
        # output_hidden_states: bool = False,
        # output_router_logits: bool = False,
        # output_expert_indexes: bool = False,
        # return_dict: bool = True,
    ) -> Union[MoEMaskedLMOutput, tuple]:
        """
        Parameters:
        -----------

        input_ids: torch.LomgTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask, of shape (batch_size, sequence_length). If boolean, ``True`` indicates that
            tokens should be ignored for attention purposes. If float, it is added to the attention
            scores.

        key_padding_mask: torch.BoolTensor
            Key padding mask. Not used (use attention_mask instead)

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
        all_self_attentions = () if self.config.output_attentions else None
        all_hidden_states = () if self.config.output_hidden_states else None
        router_logits = ()
        expert_idxs = ()
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

            if isinstance(layer, SparseTransformerLayer):
                # sparse layer, so we need to collect router/expert info
                x = layer(
                    x,
                    attention_mask=attention_mask,
                    # need_weights=self.config.output_attentions,
                    # output_router_logits=self.config.output_router_logits,
                )
                if self.config.output_attentions:
                    x, attn, router_tuple = x
                    all_self_attentions += (attn,)
                else:
                    x, router_tuple = x
                router_logits += (router_tuple[0],)
                expert_idxs += (router_tuple[1],)
            else:
                # dense layer, no router info
                x = layer(
                    x,
                    attention_mask=attention_mask,
                    # need_weights=self.config.output_attentions,
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

        # router losses
        cat_router_logits = torch.cat(router_logits, dim=1)
        cat_expert_indexes = torch.cat(expert_idxs, dim=1)
        router_probs = nn.Softmax(dim=-1)(cat_router_logits)
        z_loss = router_z_loss(cat_router_logits)
        if self.config.router_type == "expert choice":
            aux_loss = None
        else:
            aux_loss = router_load_balancing_loss(router_probs, cat_expert_indexes)

        # outputs
        if not self.config.return_dict:
            return tuple(
                v
                for v in [
                    x,
                    all_hidden_states,
                    all_self_attentions,
                    router_logits if self.config.output_router_logits else None,
                    expert_idxs if self.config.output_expert_indexes else None,
                    z_loss,
                    aux_loss,
                ]
                if v is not None
            )

        return MoEModelOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=router_logits if self.config.output_router_logits else None,
            expert_indexes=expert_idxs if self.config.output_expert_indexes else None,
            z_loss=z_loss,
            aux_loss=aux_loss,
        )

    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     inputs_embeds=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    # ):
    #     return_dict = (
    #         return_dict if return_dict is not None else self.config.use_return_dict
    #     )

    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot provide both input_ids and inputs_embeds")

    #     if input_ids is not None:
    #         input_shape = input_ids.size()
    #     else:
    #         input_shape = inputs_embeds.size()[:-1]

    #     device = input_ids.device if input_ids is not None else inputs_embeds.device

    #     if token_type_ids is None:
    #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    #     if position_ids is None and self.pos_embeddings is not None:
    #         position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
    #         position_ids = position_ids.unsqueeze(0).expand(input_shape)

    #     if inputs_embeds is None:
    #         inputs_embeds = self.embeddings(input_ids)

    #     embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
    #     if self.pos_embeddings is not None and position_ids is not None:
    #         embeddings = embeddings + self.pos_embeddings(position_ids)

    #     hidden_states = self.dropout(embeddings)

    #     if attention_mask is not None:
    #         if attention_mask.dim() == 2:
    #             attention_mask = attention_mask[:, None, None, :]
    #         attention_mask = (1.0 - attention_mask) * -10000.0

    #     all_hidden_states = () if output_hidden_states else None
    #     all_attentions = () if output_attentions else None

    #     total_aux_loss = 0.0
    #     total_z_loss = 0.0

    #     for layer in self.layers:
    #         if output_hidden_states:
    #             all_hidden_states += (hidden_states,)

    #         hidden_states, aux_loss, z_loss = layer(
    #             hidden_states, attention_mask=attention_mask
    #         )
    #         total_aux_loss += aux_loss
    #         total_z_loss += z_loss

    #     hidden_states = self.final_norm(hidden_states)

    #     if output_hidden_states:
    #         all_hidden_states += (hidden_states,)

    #     if not return_dict:
    #         return (
    #             hidden_states,
    #             all_hidden_states,
    #             all_attentions,
    #             total_aux_loss,
    #             total_z_loss,
    #         )

    #     return (
    #         BaseModelOutput(
    #             last_hidden_state=hidden_states,
    #             hidden_states=all_hidden_states,
    #             attentions=all_attentions,
    #         ),
    #         total_aux_loss,
    #         total_z_loss,
    #     )

    # class BalmMoEModel(BalmBase):
    #     """
    #     BALM Mixture-of-Experts (MoE) model.

    #     Parameters
    #     ----------
    #     config: BalmMoEConfig
    #         Configuration object defining model architecture and hyperparameters.
    #     """

    #     config_class = BalmMoEConfig

    #     def __init__(
    #         self,
    #         config: BalmMoEConfig,
    #     ):
    #         super().__init__(config)
    #         self.alternate_sparsity = self.config.alternate_sparsity

    #         # embedding
    #         self.embed_tokens = nn.Embedding(
    #             self.config.vocab_size,
    #             self.config.embed_dim,
    #             padding_idx=self.config.padding_idx,
    #         )
    #         self.embedding_dropout = nn.Dropout(self.config.token_embedding_dropout)

    #         # layers
    #         if self.config.alternate_sparsity:
    #             # alternate dense/sparse layers
    #             layers = []
    #             for layer_num in range(self.config.num_layers):
    #                 if layer_num % 2 == 0:
    #                     layers.append(
    #                         DenseTransformerLayer(
    #                             embed_dim=self.config.embed_dim,
    #                             ffn_dim=self.config.ffn_dim,
    #                             num_heads=self.config.num_heads,
    #                             max_length=self.config.max_length,
    #                             dropout=self.config.dropout,
    #                             attention_dropout=self.config.attention_dropout,
    #                             token_embedding_dropout=self.config.token_embedding_dropout,
    #                             layer_norm_eps=self.config.layer_norm_eps,
    #                             activation=self.config.activation,
    #                             positional_embedding_type=self.config.positional_embedding_type,
    #                             pre_norm=self.config.pre_norm,
    #                         )
    #                     )
    #                 else:
    #                     layers.append(
    #                         SparseTransformerLayer(
    #                             embed_dim=self.config.embed_dim,
    #                             ffn_dim=self.config.ffn_dim,
    #                             num_heads=self.config.num_heads,
    #                             max_length=self.config.max_length,
    #                             num_experts=self.config.num_experts,
    #                             expert_capacity=self.config.expert_capacity,
    #                             num_shared_experts=self.config.num_shared_experts,
    #                             send_bos_to_all_experts=self.config.send_bos_to_all_experts,
    #                             top_k=self.config.router_top_k,
    #                             dropout=self.config.dropout,
    #                             attention_dropout=self.config.attention_dropout,
    #                             expert_ffn_dropout=self.config.expert_ffn_dropout,
    #                             token_embedding_dropout=self.config.token_embedding_dropout,
    #                             layer_norm_eps=self.config.layer_norm_eps,
    #                             activation=self.config.activation,
    #                             positional_embedding_type=self.config.positional_embedding_type,
    #                             pre_norm=self.config.pre_norm,
    #                             router_dtype=self.config.router_dtype,
    #                             router_bias=self.config.router_bias,
    #                             router_jitter=self.config.router_jitter,
    #                             router_ignore_padding_tokens=self.config.router_ignore_padding_tokens,
    #                             expert_choice_router=self.config.expert_choice_router,
    #                         )
    #                     )
    #             self.layers = nn.ModuleList(layers)
    #         else:
    #             # all sparse layers
    #             self.layers = nn.ModuleList(
    #                 [
    #                     SparseTransformerLayer(
    #                         embed_dim=self.config.embed_dim,
    #                         ffn_dim=self.config.ffn_dim,
    #                         num_heads=self.config.num_heads,
    #                         max_length=self.config.max_length,
    #                         num_experts=self.config.num_experts,
    #                         expert_capacity=self.config.expert_capacity,
    #                         num_shared_experts=self.config.num_shared_experts,
    #                         send_bos_to_all_experts=self.config.send_bos_to_all_experts,
    #                         top_k=self.config.router_top_k,
    #                         dropout=self.config.dropout,
    #                         attention_dropout=self.config.attention_dropout,
    #                         expert_ffn_dropout=self.config.expert_ffn_dropout,
    #                         token_embedding_dropout=self.config.token_embedding_dropout,
    #                         layer_norm_eps=self.config.layer_norm_eps,
    #                         activation=self.config.activation,
    #                         positional_embedding_type=self.config.positional_embedding_type,
    #                         pre_norm=self.config.pre_norm,
    #                         router_dtype=self.config.router_dtype,
    #                         router_bias=self.config.router_bias,
    #                         router_jitter=self.config.router_jitter,
    #                         router_ignore_padding_tokens=self.config.router_ignore_padding_tokens,
    #                         expert_choice_router=self.config.expert_choice_router,
    #                     )
    #                     for _ in range(self.config.num_layers)
    #                 ]
    #             )

    #         # final layer norm
    #         self.final_norm = nn.LayerNorm(self.config.embed_dim)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     key_padding_mask: Optional[torch.Tensor] = None,
    #     output_attentions: bool = False,
    #     output_hidden_states: bool = False,
    #     output_router_logits: bool = False,
    #     output_expert_indexes: bool = False,
    #     return_dict: bool = True,
    # ) -> Union[MoEMaskedLMOutput, tuple]:
    #     """
    #     Parameters:
    #     -----------

    #     input_ids: torch.LomgTensor
    #         Tokenized input IDs

    #     attention_mask: torch.BoolTensor
    #         Attention mask

    #     key_padding_mask: torch.BoolTensor
    #         Key padding mask

    #     output_attentions: bool
    #         Whether to output attention weights

    #     output_hidden_states: bool
    #         Whether to output hidden states

    #     output_router_logits: bool
    #         Whether to output router logits

    #     output_expert_indexes: bool
    #         Whether to output expert indexes

    #     return_dict: bool
    #         Whether to return a dictionary of outputs (returns a tuple if False)

    #     Returns:
    #     --------
    #     output (tuple or dict):
    #         If `return_dict` is ``True``, the output is a ``MoEModelOutput`` object:
    #             - last_hidden_state (torch.FloatTensor): last hidden state
    #             - z_loss (torch.FloatTensor): router z loss
    #             - aux_loss (torch.FloatTensor): router auxiliary loss
    #             - hidden_states (torch.FloatTensor): hidden states
    #             - attentions (torch.FloatTensor): attention weights
    #             - router_logits (torch.FloatTensor): router logits
    #             - expert_indexes (torch.LongTensor): expert indexes

    #         If `return_dict` is ``False``, the output is a ``tuple`` with the f0llowing elements:
    #             - last_hidden_state (torch.FloatTensor): last hidden state
    #             - z_loss (torch.FloatTensor): router z loss
    #             - aux_loss (torch.FloatTensor): router auxiliary loss
    #             - hidden_states (torch.FloatTensor): hidden states
    #             - attentions (torch.FloatTensor): attention weights
    #             - router_logits (torch.FloatTensor): router logits
    #             - expert_indexes (torch.LongTensor): expert indexes

    #         For attentions, hidden_states, router_logits, and expert_indexes, if they are not output, the corresponding
    #         value will be ``None`` (for ``MoEModelOutput``) or not returned at all (for ``tuple``).

    #     """
    #     # init
    #     all_self_attentions = () if output_attentions else None
    #     all_hidden_states = () if output_hidden_states else None
    #     router_logits = ()
    #     expert_indexes = ()

    #     # embeddings
    #     x = self.embed_tokens(x)

    #     # layers
    #     for layer in self.layers:
    #         if output_hidden_states:
    #             all_hidden_states = all_hidden_states + (x,)

    #         if isinstance(layer, SparseTransformerLayer):
    #             # sparse layer, so we need to collect router/expert info
    #             x = layer(
    #                 x,
    #                 attention_mask=attention_mask,
    #                 key_padding_mask=key_padding_mask,
    #                 need_weights=output_attentions,
    #                 # output_router_logits=output_router_logits,
    #             )
    #             if output_attentions:
    #                 x, attn, router_tuple = x
    #                 all_self_attentions = all_self_attentions + (attn,)
    #             else:
    #                 x, router_tuple = x

    #             router_logits = router_logits + (router_tuple[0],)
    #             expert_indexes = expert_indexes + (router_tuple[1],)
    #         else:
    #             # dense layer, no router info needed
    #             x = layer(
    #                 x,
    #                 attention_mask=attention_mask,
    #                 need_weights=output_attentions,
    #             )
    #             if output_attentions:
    #                 x, attn = x
    #                 all_self_attentions = all_self_attentions + (attn,)

    #     # final layer norm
    #     if self.config.pre_norm:
    #         x = self.final_norm(x)

    #     # save the last hidden state
    #     if output_hidden_states:
    #         all_hidden_states = all_hidden_states + (x,)

    #     # router losses
    #     cat_router_logits = torch.cat(router_logits, dim=1)
    #     cat_expert_indexes = torch.cat(expert_indexes, dim=1)
    #     router_probs = nn.Softmax(dim=-1)(cat_router_logits)
    #     z_loss = router_z_loss(cat_router_logits)
    #     if self.config.expert_choice_router:
    #         aux_loss = None
    #     else:
    #         aux_loss = router_load_balancing_loss(router_probs, cat_expert_indexes)

    #     # outputs
    #     if not return_dict:
    #         return tuple(
    #             v
    #             for v in [
    #                 x,
    #                 all_hidden_states,
    #                 all_self_attentions,
    #                 router_logits if output_router_logits else None,
    #                 expert_indexes if output_expert_indexes else None,
    #                 z_loss,
    #                 aux_loss,
    #             ]
    #             if v is not None
    #         )

    #     return MoEModelOutput(
    #         last_hidden_state=x,
    #         z_loss=z_loss,
    #         aux_loss=aux_loss,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attentions,
    #         router_logits=router_logits if output_router_logits else None,
    #         expert_indexes=expert_indexes if output_expert_indexes else None,
    #     )

    #     # # results
    #     # result = MaskedLMOutput(
    #     #     last_hidden_state=x,
    #     #     router_z_loss=z_loss,
    #     #     router_aux_loss=aux_loss,
    #     # )
    #     # if need_weights:
    #     #     # attentions: B x L x H x T x T
    #     #     attentions = torch.stack(attn_weights, 1)
    #     #     attentions = attentions * attention_mask[:, None, None, :, :]
    #     #     result["attentions"] = attentions
    #     # if output_hidden_states:
    #     #     result["hidden_states"] = hidden_states
    #     # if output_router_logits:
    #     #     result["router_logits"] = cat_router_logits
    #     # if output_expert_indexes:
    #     #     result["expert_indexes"] = cat_expert_indexes
    #     # if return_dict:
    #     #     return result
    #     # return result.as_tuple()


class BalmMoEForMaskedLM(PreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin):
    """
    BALM Mixture-of-Experts (MoE) model for masked language modeling.
    Uses the BALM-MoE encoder and adds a masked language modeling head.

    Parameters
    ----------
    config: BalmMoEConfig
        Configuration object defining model architecture and hyperparameters.

    """

    config_class = BalmMoEConfig
    base_model_prefix = "balm"

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
        super().__init__(config)
        # model
        self.balm = BalmMoEModel(config=self.config)

        # LM head
        self.lm_head = BalmLMHead(
            hidden_size=self.config.hidden_size,
            output_dim=self.config.vocab_size,
            activation=self.config.classifier_activation,
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # router loss coefficients
        # set as attributes (rather than using the config values directly)
        # so that we can zero them out in freeze_base_model() if necessary
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        # output_attentions: bool = False,
        # output_hidden_states: bool = False,
        # output_router_logits: bool = False,
        # output_expert_indexes: bool = False,
        return_dict: Optional[bool] = None,
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
            Key padding mask. Not used (use attention_mask instead)

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
            # key_padding_mask=key_padding_mask,
            # output_attentions=self.config.output_attentions,
            # output_hidden_states=self.config.output_hidden_states,
            # output_router_logits=self.config.output_router_logits,
            # output_expert_indexes=self.config.output_expert_indexes,
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
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

            # router loss(es)
            z_loss = self.router_z_loss_coef * (outputs.z_loss)
            if self.config.router_type == "expert choice":
                loss = lm_loss + z_loss
                aux_loss = None
            else:
                aux_loss = self.router_aux_loss_coef * (outputs.aux_loss)
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
                    lm_logits,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits,
                    outputs.expert_indexes,
                    aux_loss,
                    z_loss,
                    lm_loss,
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
            z_loss=z_loss,
            aux_loss=aux_loss,
            lm_loss=lm_loss,
        )

        # # outputs
        # if return_dict:
        #     return outputs.as_dict()
        # return outputs.as_tuple()


class BalmMoEForSequenceClassification(
    PreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin
):
    """
    BALM Mixture-of-Experts (MoE) model for sequence classification.
    Uses the BALM-MoE encoder and adds a sequence-level classification head.

    Parameters
    ----------
    config : BalmMoEConfig
        Configuration object defining model architecture and hyperparameters.

    """

    config_class = BalmMoEConfig
    base_model_prefix = "balm"

    def __init__(
        self,
        config: BalmMoEConfig,
    ):
        super().__init__(config)
        # model
        self.balm = BalmMoEModel(config=self.config)

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

        # router loss coefficients
        self.router_z_loss_coef = self.config.router_z_loss_coef
        self.router_aux_loss_coef = self.config.router_aux_loss_coef

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        # output_attentions: bool = False,
        # output_hidden_states: bool = False,
        # output_router_logits: bool = False,
        # output_expert_indexes: bool = False,
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
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # output_router_logits=output_router_logits,
            # output_expert_indexes=output_expert_indexes,
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
            z_loss = self.router_z_loss_coef * (outputs.z_loss)
            if self.config.router_type == "expert choice":
                loss = classifier_loss + z_loss
                aux_loss = None
            else:
                aux_loss = self.router_aux_loss_coef * (outputs.aux_loss)
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
                    classifier_logits,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.router_logits,
                    outputs.expert_indexes,
                    aux_loss,
                    z_loss,
                    classifier_loss,
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
            z_loss=z_loss,
            aux_loss=aux_loss,
            classifier_loss=classifier_loss,
        )

        # outputs = self.balm(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     key_padding_mask=key_padding_mask,
        #     need_weights=need_weights,
        # )
        # x = outputs["last_hidden_state"]

        # classifier_logits = self.classifier(x)
        # outputs["logits"] = classifier_logits

        # if labels is not None:
        #     labels = labels.to(classifier_logits.device)
        #     loss = self.criterion(classifier_logits, labels)
        #     outputs["loss"] = loss

        # if return_dict:
        #     return outputs.as_dict()
        # return outputs.as_tuple()

        # x = self.balm(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     key_padding_mask=key_padding_mask,
        #     need_weights=output_attentions,
        # )
        # if output_attentions:
        #     x, attn = x
        # logits = self.classifier(x)

        # classifier_loss = None
        # if labels is not None:
        #     classifier_loss = self.criterion(
        #         logits.view(-1, logits.size(-1)),
        #         labels.view(-1),
        #     )

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
