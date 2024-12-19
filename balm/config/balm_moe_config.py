# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Optional, Union

from transformers import PretrainedConfig

# from .base import BaseConfig


class BalmMoEConfig(PretrainedConfig):
    """
    Configuration for BalmMoE models.

    Parameters
    ----------
    vocab_size : int, default=32
        The vocabulary size of the model.

    hidden_size : int, default=320
        The hidden size of the model.

    num_hidden_layers : int, default=6
        The number of hidden layers in the model.

    num_attention_heads : int, default=20
        The number of attention heads in the model.

    intermediate_size : int, default=1280
        The intermediate size of the model.

    activation : str, default="swiglu"
        The activation function to use for the model.

    dropout : float, default=0.1
        The dropout probability for the model. Can be overridden
        by `attention_dropout`, `hidden_dropout`, and `expert_dropout`.

    attention_dropout : float, default=0.1
        The dropout probability for the attention layers.

    hidden_dropout : float, default=0.1
        The dropout probability for the hidden layers.

    max_position_embeddings : int, default=320
        The maximum position embeddings.

    initializer_range : float, default=0.02
        The initializer range for the model.

    layer_norm_eps : float, default=1e-12
        The epsilon for layer normalization.

    position_embedding_type : str, default="rotary"
        The type of position embeddings to use.
        Options are "rotary" or "absolute".

    pre_norm : bool, default=True
        Whether to use pre-normalization.

    token_dropout : float, default=0.0
        The dropout probability for the token embeddings.

    mask_token_id : int, default=31
        The mask token id.

    pad_token_id : int, default=1
        The pad token id.

    type_vocab_size : int, default=2
        Vocabulary size of the token_type_ids`

    num_experts : int, default=4
        The number of experts in the model.

    top_k : int, default=1
        The top k to use for the router.

    router_type : str, default="top-k"
        The type of router to use.
        Options are "top-k" or "expert-choice".

    expert_capacity_type : str, default="multiplier"
        The type of expert capacity to use.
        If "absolute": tokens per expert; if "multiplier": capacity = multiplier * max_position_embeddings

    expert_capacity : int, default=2
        The capacity of each expert.
        If `expert_capacity_type` is "absolute", this value is translated as the actual token capacity of each expert.
        If `expert_capacity_type` is "multiplier", this value is translated as the multiplier with which the total
        expert capacity is calculated (i.e. each expert capacity = multiplier * max_position_embeddings / num_experts).

    expert_activation : str, default="gelu"
        The activation function to use for the experts.
        Options are "swiglu", "relu", "gelu".

    expert_bias : bool, default=True
        Whether to use a bias for the experts.

    expert_dropout : float, default=0.1
        The dropout probability for the expert layers.

    alternate_sparsity : bool, default=True
        Whether to use alternate sparse and dense layers.

    aux_loss_coef : float, default=0.01
        The coefficient for the auxiliary loss.

    z_loss_coef : float, default=0.001
        The coefficient for the z-loss.

    num_labels : int, default=2
        The number of labels for the classification head (sequence or token classification).

    num_choices : int, default=4
        The number of choices for the multiple choice classification head.

    output_attentions : bool, default=False
        Whether to output the attentions.

        .. warning::
            If ``output_attentions`` is ``True``, torch can't use optimized SDPA.
            See `here`_ for more details.

    output_hidden_states : bool, default=False
        Whether to output the hidden states.

    use_cache : bool, default=True
        Whether to use the cache.

    **kwargs : dict, optional
        Additional keyword arguments are passed to the parent class (transformers.PretrainedConfig).

    Raises
    ------
    ValueError
        If the positional embedding type is not valid.

    ValueError
        If the router type is not valid.

    ValueError
        If the expert capacity type is not valid.

    ValueError
        If the FFN or expert activation functions are not valid.

    .. _here:
        https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward

    """

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 320,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 20,
        intermediate_size: Optional[int] = None,
        activation: str = "swiglu",  # "swiglu", "relu", "gelu"
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        hidden_dropout: Optional[float] = None,
        max_position_embeddings: int = 320,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "rotary",
        pre_norm: bool = True,
        token_dropout: float = 0.0,
        mask_token_id: int = 31,
        pad_token_id: int = 1,
        type_vocab_size: int = 2,
        # MoE params
        num_experts: int = 8,
        num_shared_experts: int = 0,
        top_k: int = 1,
        router_type: str = "topk",  # "topk" or "expert choice"
        router_dtype: str = "float32",
        router_jitter: float = 0.0,
        router_bias: bool = False,
        expert_capacity_type: str = "multiplier",  # "absolute" or "multiplier"
        expert_capacity: Union[int, float] = 2,
        expert_activation: str = "gelu",  # "swiglu", "relu", "gelu"
        expert_bias: bool = True,
        expert_dropout: Optional[float] = None,
        alternate_sparsity: bool = True,
        # router losses
        router_aux_loss_coef: float = 0.01,  # coefficient for aux loss (load balancing, top-k only)
        router_z_loss_coef: float = 0.001,  # coefficient for z-loss
        # classification
        classifier_activation: str = "tanh",  # "swiglu", "relu", "gelu", or "tanh"
        num_labels: int = 2,  # sequence/token-level classification
        num_choices: int = 4,  # multiple choice classification
        # outputs
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        output_expert_indexes: bool = False,
        return_dict: bool = True,
        # 🤗 integration
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs
        )

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size or hidden_size * 4)
        self.activation = activation.lower()
        self.dropout = float(dropout)
        self.attention_dropout = float(
            attention_dropout if attention_dropout is not None else dropout
        )
        self.hidden_dropout = float(
            hidden_dropout if hidden_dropout is not None else dropout
        )
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.position_embedding_type = position_embedding_type.lower()
        self.pre_norm = bool(pre_norm)
        self.token_dropout = float(token_dropout)
        self.type_vocab_size = int(type_vocab_size)
        self.use_cache = bool(use_cache)

        # classification
        self.num_labels = int(num_labels)
        self.num_choices = int(num_choices)
        self.classifier_activation = classifier_activation.lower()

        # outputs
        self.output_attentions = bool(output_attentions)
        self.output_hidden_states = bool(output_hidden_states)
        self.output_router_logits = bool(output_router_logits)
        self.output_expert_indexes = bool(output_expert_indexes)

        # MoE params
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.top_k = int(top_k)
        self.router_type = self._standardize_router_type(router_type)
        self.router_dtype = router_dtype.lower()
        self.router_jitter = float(router_jitter)
        self.router_bias = bool(router_bias)
        self.expert_capacity_type = expert_capacity_type.lower()
        self.expert_capacity = expert_capacity
        self.expert_activation = expert_activation.lower()
        self.expert_bias = bool(expert_bias)
        self.expert_dropout = float(
            expert_dropout if expert_dropout is not None else dropout
        )
        self.alternate_sparsity = bool(alternate_sparsity)

        # router losses
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.router_z_loss_coef = float(router_z_loss_coef)

        # validate params
        if self.position_embedding_type not in ["rotary", "relative", "absolute"]:
            raise ValueError(
                f"Invalid positional embedding type: {self.position_embedding_type}. Options are 'rotary', 'relative', or 'absolute'."
            )
        if self.expert_capacity_type not in ["absolute", "multiplier"]:
            raise ValueError(
                f"Invalid expert capacity type: {self.expert_capacity_type}. Options are 'absolute' or 'multiplier'."
            )
        if self.activation not in ["swiglu", "relu", "gelu"]:
            raise ValueError(
                f"Invalid FFN activation: {self.activation}. Options are 'swiglu', 'relu', or 'gelu'."
            )
        if self.expert_activation not in ["swiglu", "relu", "gelu"]:
            raise ValueError(
                f"Invalid expert activation: {self.expert_activation}. Options are 'swiglu', 'relu', or 'gelu'."
            )
        if self.classifier_activation not in ["swiglu", "relu", "gelu", "tanh"]:
            raise ValueError(
                f"Invalid classifier activation: {self.classifier_activation}. Options are 'swiglu', 'relu', 'gelu', or 'tanh'."
            )

    def _standardize_router_type(self, router_type: str) -> str:
        if router_type.lower() in ["topk", "top-k", "top_k"]:
            return "topk"
        elif router_type.lower() in ["expert choice", "expert-choice", "expert_choice"]:
            return "expert choice"
        else:
            raise ValueError(
                f"Invalid router type: {router_type}. Options are 'topk' or 'expert choice'."
            )


# @dataclass
# class SparseMoETransformerConfig(PretrainedConfig):
#     """
#     Configuration for Sparse Mixture-of-Experts Transformer.
#     """

#     vocab_size: int = 32
#     hidden_size: int = 320
#     num_hidden_layers: int = 6
#     num_attention_heads: int = 20
#     intermediate_size: int = 1280
#     hidden_act: str = "swiglu"
#     max_position_embeddings: int = 320
#     rotary_embeddings: bool = True
#     relative_positions: bool = False
#     type_vocab_size: int = 2
#     layer_norm_eps: float = 1e-12
#     dropout_prob: float = 0.1
#     pad_token_id: int = 0
#     bos_token_id: int = 101
#     eos_token_id: int = 102

#     # MoE params
#     num_experts: int = 4
#     top_k: int = 1  # for top-k routing
#     router_type: str = "top-k"  # "top-k" or "expert-choice"
#     expert_capacity_type: str = "multiplier"  # "absolute" or "multiplier"
#     expert_capacity: int = 2  # if "absolute": tokens per expert; if "multiplier": capacity = multiplier * max_seq_len
#     expert_activation: str = "gelu"  # "swiglu", "relu", "gelu"

#     # router losses
#     aux_loss_coef: float = 0.01  # coefficient for aux loss (load balancing, top-k only)
#     z_loss_coef: float = 0.001  # coefficient for z-loss

#     # sequence classification
#     num_labels: int = 2
#     # token classification
#     num_labels_token_classification: int = 9
#     # multiple choice classification
#     num_choices: int = 4

#     # model initialization range
#     initializer_range: float = 0.02

#     # for huggingface integration
#     output_attentions: bool = False
#     output_hidden_states: bool = False
#     use_cache: bool = True


# class BalmMoEConfig(BaseConfig):
#     def __init__(
#         self,
#         embed_dim: int = 320,
#         ffn_dim: int = 1280,
#         num_layers: int = 6,
#         num_heads: int = 20,
#         num_experts: int = 8,
#         max_length: int = 320,
#         vocab_size: int = 33,
#         expert_capacity: Optional[int] = None,
#         expert_capacity_multiplier: float = 1.5,
#         num_shared_experts: int = 0,
#         send_bos_to_all_experts: bool = True,
#         activation: str = "swiglu",
#         positional_embedding_type: str = "rotary",
#         pre_norm: bool = True,
#         router_z_loss_coef: float = 1e-3,
#         router_aux_loss_coef: float = 1e-2,
#         alternate_sparsity: bool = False,
#         router_top_k: int = 1,
#         router_bias: bool = False,
#         router_jitter: float = 0.0,
#         router_dtype: str = "float32",
#         router_ignore_padding_tokens: bool = True,
#         expert_choice_router: bool = False,
#         dropout: float = 0.1,
#         attention_dropout: float = 0.0,
#         expert_ffn_dropout: float = 0.0,
#         token_embedding_dropout: float = 0.0,
#         layer_norm_eps: float = 1e-5,
#         padding_idx: int = 0,
#         # classification head
#         num_labels: int = 2,
#         classifier_dropout: float = 0.0,
#         classifier_activation: str = "tanh",
#     ):
#         """
#         Configuration for the BalmMoE model. Default parameters are similar to the 8M parameter ESM-2 model.

#         Parameters
#         ----------
#         embed_dim : int, default=320
#             The dimension of the token embeddings.

#         ffn_dim : int, default=1280
#             The dimension of the feed-forward network.

#         num_layers : int, default=6
#             The number of layers in the transformer.

#         num_heads : int, default=20
#             The number of heads in the transformer.

#         num_experts : int, default=8
#             The number of experts in the transformer.

#         max_length : int, default=320
#             The maximum length of the input sequence.

#         vocab_size : int, default=33
#             The vocabulary size.

#         expert_capacity : int, optional
#             The capacity of each expert. If not provided, it will be calculated as `max_length / num_experts * expert_capacity_multiplier`.

#         expert_capacity_multiplier : float, default=1.25
#             The multiplier for the expert capacity.

#         num_shared_experts : int, default=0
#             The number of shared experts in the transformer.

#         send_bos_to_all_experts : bool, default=True
#             Whether to send the BOS token to all experts. The effective expert capacity will be reduced by one if `send_bos_to_all_experts` is True.

#         activation : str, default="gelu"
#             The activation function to use for the experts. Options are "relu" and "gelu".

#         router_z_loss_coef : float, default=0.001
#             The coefficient for the router z loss.

#         router_aux_loss_coef : float, default=0.001
#             The coefficient for the router auxiliary loss.

#         alternate_sparsity : bool, default=False
#             Whether to use alternate sparsity for the router.

#         router_top_k : int, default=1
#             The top k to use for the router.

#         router_bias : bool, default=False
#             Whether to use a bias for the router.

#         router_jitter : float, default=0.0
#             The jitter to use for the router.

#         router_dtype : str, default="float32"
#             The dtype to use for the router. Options are "float32" and "float16".

#         router_ignore_padding_tokens : bool, default=True
#             Whether to ignore padding tokens for the router.

#         dropout : float, default=0.1
#             The dropout to use for the transformer.

#         attention_dropout : float, default=0.0
#             The dropout to use for the attention.

#         ffn_dropout : float, default=0.0
#             The dropout to use for the experts.

#         token_embedding_dropout : float, default=0.0
#             The dropout to use for the token embeddings.

#         layer_norm_eps : float, default=1e-5
#             The epsilon to use for the layer normalization.

#         padding_idx : int, default=0
#             The index to use for the padding tokens.

#         num_labels : int, default=2
#             The number of labels for the classification head.

#         classifier_dropout : float, default=0.0
#             The dropout to use for the classification head.

#         classifier_activation : str, default="tanh"
#             The activation function to use for the classification head. Options are "tanh" and "softmax".
#         """
#         super().__init__()
#         self.embed_dim = int(embed_dim)
#         self.ffn_dim = int(ffn_dim)
#         self.num_layers = int(num_layers)
#         self.num_heads = int(num_heads)
#         self.num_experts = int(num_experts)
#         self.max_length = int(max_length)
#         self.vocab_size = int(vocab_size)
#         self.expert_capacity_multiplier = float(expert_capacity_multiplier)
#         self.expert_capacity = (
#             int(expert_capacity)
#             if expert_capacity is not None
#             else int(max_length / num_experts * self.expert_capacity_multiplier)
#         )
#         self.num_shared_experts = int(num_shared_experts)
#         self.send_bos_to_all_experts = bool(send_bos_to_all_experts)
#         if positional_embedding_type.lower() not in ["rotary", "relative"]:
#             raise ValueError(
#                 f"Invalid positional embedding type: {positional_embedding_type}. Options are 'rotary' or 'relative'."
#             )
#         self.positional_embedding_type = positional_embedding_type.lower()
#         if activation.lower() not in ["swiglu", "relu", "gelu"]:
#             raise ValueError(
#                 f"Invalid FFN activation: {activation}. Options are 'swiglu', 'relu', or 'gelu'."
#             )
#         self.activation = activation.lower()
#         self.pre_norm = bool(pre_norm)
#         self.router_z_loss_coef = float(router_z_loss_coef)
#         self.router_aux_loss_coef = float(router_aux_loss_coef)
#         self.alternate_sparsity = alternate_sparsity
#         self.router_top_k = int(router_top_k)
#         self.router_bias = bool(router_bias)
#         self.router_jitter = float(router_jitter)
#         self.router_dtype = router_dtype.lower()
#         self.router_ignore_padding_tokens = bool(router_ignore_padding_tokens)
#         self.expert_choice_router = bool(expert_choice_router)
#         self.dropout = float(dropout)
#         self.attention_dropout = float(attention_dropout)
#         self.expert_ffn_dropout = float(expert_ffn_dropout)
#         self.token_embedding_dropout = float(token_embedding_dropout)
#         self.layer_norm_eps = float(layer_norm_eps)
#         self.padding_idx = int(padding_idx)

#         # classification head
#         self.num_labels = int(num_labels)
#         self.classifier_dropout = float(classifier_dropout)
#         if classifier_activation.lower() not in ["tanh", "relu", "gelu"]:
#             raise ValueError(
#                 f"Invalid classification head activation: {classifier_activation}. Options are 'tanh', 'relu', or 'gelu'."
#             )
#         self.classifier_activation = classifier_activation.lower()
