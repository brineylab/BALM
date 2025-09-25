# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union, List, Literal

from transformers import PretrainedConfig


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
    intermediate_size : int, default=None
        The intermediate size of the model.
        If not provided, defaults to (`hidden_size` * 4).
    activation : {"gelu", "relu", "glu", "swiglu", "geglu", "reglu"}, default="swiglu"
        The activation function to use for the model.
    dropout : float, default=0.1
        The dropout probability for the model. Can be overridden
        by `attention_dropout`, `hidden_dropout`, and `expert_dropout`.
    attention_dropout : float, default=None
        The dropout probability for the attention layers.
        If not provided, defaults to `dropout`.
    hidden_dropout : float, default=None
        The dropout probability for the hidden layers.
        If not provided, defaults to `dropout`.
    ffn_bias : bool, default=True
        Whether to use a bias for FFN layers. Used in DenseTransformerLayers only.
    max_position_embeddings : int, default=256
        The maximum position embeddings.
    initializer_range : float, default=0.02
        The initializer range for the model.
    layer_norm_eps : float, default=1e-5
        The epsilon for layer normalization.
    position_embedding_type : {"rotary", "absolute"}, default="rotary"
        The type of position embeddings to use.
    mask_token_id : int, default=31
        The mask token id.
    pad_token_id : int, default=1
        The pad token id.
    num_experts : int, default=8
        The total number of experts in the model.
        If there are any shared experts, these experts are included in this total.
    num_shared_experts: int, default=0
        The number of shared experts (which receive all tokens) in the model.
    num_experts_per_tok : int, default=1
        The number of experts to route each token to. Only used if `router_type` is "top-k".
    top_p_threshold: float, default=0.7
        The probability threshold for "top-p" routing. Only used if `router_type` is "top-p".
    num_initial_dense_layers: int, default=1
        The number of dense layers at the start of the model before any sparse layers.
    alternate_sparsity : bool, default=True
        Whether to use alternate sparse and dense layers.
    router_type : {"top-k", "top-p", "expert-choice"}, default="top-k"
        The type of router to use.
    router_dtype : {"float32", "float16", "bfloat16"}, default="float32"
        Data type of the router tensors, that is converted to torch.dtype.
    router_jitter: float, default=0.0
        Jitter to apply to inputs of the router.
    router_bias: bool, default=False
        Whether to use a bias in the router.
    router_aux_loss_coef: float, default=0.01
        The coefficient for the auxiliary loss.
    router_mask_aux_loss: bool, default=False
        Whether to exclude padding tokens in the auxiliary loss.
    router_z_loss_coef: float, default=0.001
        The coefficient for the z-loss.
    router_use_penalty_loss: bool, default=False
        Whether or not to use p-penalty loss instead of standard aux loss.
    router_penalty_loss_coef: float, default=0.1
        The coefficient for the p-penalty loss for heterogeneous sized experts.
    router_dynamic_loss_coef: float, default=0.0001
        The coefficient for the dynamic loss for top-p models.
    expert_capacity_type : {"absolute", "multiplier"}, default="multiplier"
        The type of expert capacity to use.
        If "absolute": tokens per expert; if "multiplier": capacity = multiplier * max_position_embeddings
    expert_capacity: int, default=1
        The capacity of each expert, excluding shared experts.
        If `expert_capacity_type` is "absolute", this value is translated as the actual token capacity of each expert.
        If `expert_capacity_type` is "multiplier", this value is translated as the multiplier with which the total
        expert capacity is calculated (i.e. each expert capacity = multiplier * max_position_embeddings / num_experts).
        If capacity is less than 0, no expert capacity is set. This is only compatible with Top-K routing.
    expert_intermediate_size: Union[int, List], default=None
        The intermediate size of the experts.
        If provided as an int, all experts will have the same size (homogeneous).
        If provided as a list, expert sizes will correspond to the sizes in the list (heterogeneous)
        If not provided, defaults to the intermediate_size.
    shared_expert_intermediate_size: int, default=None
        The intermediate size of the experts.
        If not provided:
            - If `expert_intermediate_size` is an integer, defaults to `expert_intermediate_size`.
            - If `expert_intermediate_size` is a list, defaults to the `intermediate_size`.
    expert_activation : {"gelu", "relu", "glu", "swiglu", "geglu", "reglu"}, default="swiglu"
        The activation function to use for the experts.
    expert_dropout : float
        The dropout probability for the expert layers.
    expert_bias : bool, default=True
        Whether to use a bias in expert FFN layers. Used in SparseTransformerLayers only.
    mlm_activation: {"tanh", "relu", "gelu"}, default="gelu"
        The activation function to use for the LM head.
    attention_classifier: bool, default=False
        Whether to add attention to classification head.
    classifier_attention_heads: int, default=None
        Number of attention heads in the classifier.
        If not provided, defaults to `num_attention_heads`.
        Only used if `attention_classifier` is True.
    classifier_activation: {"tanh", "relu", "gelu"} or None, default=None
        The activation function to use for the classifier. If None, defaults to
        "relu" when `attention_classifier` is True and "tanh" otherwise.
    classifier_freeze_base: bool, default=True
        Whether to freeze the base weights of classification model.
    num_labels : int, default=2
        The number of labels for the classification head (sequence or token classification).
    output_classifier_attentions : bool, default=False
        Whether to output the classifier attention.
        Only used if `attention_classifier` is True.
    output_attentions : bool, default=False
        Whether to output the attentions.
    output_hidden_states : bool, default=False
        Whether to output the hidden states.
    output_router_logits: bool, default=False
        Whether to output router logits.
    output_expert_indexes: bool, default=False
        Whether to output expert indexes.
    return_dict: bool, default = True
        Whether to return a dictionary of outputs (returns a tuple if False).
    use_cache : bool, default=True
        Whether to use the cache.
    **kwargs : dict, optional
        Additional keyword arguments are passed directly to the parent class (`transformers.PretrainedConfig`).
    """

    model_type = "balm_moe"

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 320,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 20,
        intermediate_size: Optional[int] = None,
        activation: Literal[
            "gelu", "relu", "glu", "swiglu", "geglu", "reglu"
        ] = "swiglu",
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        hidden_dropout: Optional[float] = None,
        ffn_bias: bool = True,
        max_position_embeddings: int = 256,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        position_embedding_type: Literal["rotary", "absolute"] = "rotary",
        mask_token_id: int = 31,
        pad_token_id: int = 1,
        ## MoE params
        num_experts: int = 8,
        num_shared_experts: int = 0,
        num_experts_per_tok: int = 1,  # k for top-k routing (to comply with 🤗 naming)
        top_p_threshold: float = 0.7,  # threshold for top-p routing
        num_initial_dense_layers: int = 1,
        alternate_sparsity: bool = True,
        # router
        router_type: Literal["top-k", "top-p", "expert-choice"] = "top-k",
        router_dtype: Literal["float32", "float16", "bfloat16"] = "float32",
        router_jitter: float = 0.0,
        router_bias: bool = False,
        # router losses
        router_aux_loss_coef: float = 0.01,
        router_mask_aux_loss: bool = False,
        router_z_loss_coef: float = 0.001,
        router_use_penalty_loss: bool = False,
        router_penalty_loss_coef: float = 0.1,
        router_dynamic_loss_coef: float = 0.0001,
        # experts
        expert_capacity_type: Literal["absolute", "multiplier"] = "multiplier",
        expert_capacity: int = 1,
        expert_intermediate_size: Optional[Union[int, List[int]]] = None,
        shared_expert_intermediate_size: Optional[int] = None,
        expert_activation: Literal[
            "gelu", "relu", "glu", "swiglu", "geglu", "reglu"
        ] = "swiglu",
        expert_dropout: Optional[float] = None,
        expert_bias: bool = True,
        # mlm
        mlm_activation: Literal["tanh", "relu", "gelu"] = "gelu",
        # classification
        attention_classifier: bool = False,
        classifier_attention_heads: Optional[int] = None,
        classifier_activation: Optional[Literal["tanh", "relu", "gelu"]] = None,
        classifier_freeze_base: bool = True,
        num_labels: int = 2,  # sequence/token-level classification
        output_classifier_attentions: bool = False,
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
        self.ffn_bias = bool(ffn_bias)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.position_embedding_type = position_embedding_type.lower()

        # MoE params
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.top_p_threshold = float(top_p_threshold)
        self.num_initial_dense_layers = int(num_initial_dense_layers)
        self.alternate_sparsity = bool(alternate_sparsity)
        self.router_type = self._standardize_router_type(router_type)
        self.router_dtype = router_dtype.lower()
        self.router_jitter = float(router_jitter)
        self.router_bias = bool(router_bias)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.router_mask_aux_loss = bool(router_mask_aux_loss)
        self.router_z_loss_coef = float(router_z_loss_coef)
        self.router_use_penalty_loss = bool(router_use_penalty_loss)
        self.router_penalty_loss_coef = float(router_penalty_loss_coef)
        self.router_dynamic_loss_coef = float(router_dynamic_loss_coef)
        self.expert_capacity_type = expert_capacity_type.lower()
        self.expert_capacity = expert_capacity
        self.expert_intermediate_size = self._get_expert_sizes(
            expert_intermediate_size=expert_intermediate_size
        )
        self.shared_expert_intermediate_size = self._get_shared_expert_size(
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            expert_intermediate_size=expert_intermediate_size,
        )
        self.expert_activation = expert_activation.lower()
        self.expert_dropout = float(
            expert_dropout if expert_dropout is not None else dropout
        )
        self.expert_bias = bool(expert_bias)

        # mlm
        self.mlm_activation = mlm_activation.lower()

        # classification
        self.attention_classifier = bool(attention_classifier)
        self.classifier_attention_heads = int(
            classifier_attention_heads
            if classifier_attention_heads is not None
            else num_attention_heads
        )
        self.classifier_activation = self._get_classifier_activation(
            activation=classifier_activation, use_attention=self.attention_classifier
        )
        self.classifier_freeze_base = bool(classifier_freeze_base)
        self.num_labels = int(num_labels)
        self.output_classifier_attentions = bool(output_classifier_attentions)

        # outputs
        self.output_attentions = bool(output_attentions)
        self.output_hidden_states = bool(output_hidden_states)
        self.output_router_logits = bool(output_router_logits)
        self.output_expert_indexes = bool(output_expert_indexes)
        self.return_dict = bool(return_dict)

        # 🤗 integration
        self.use_cache = bool(use_cache)

        # validate params
        if self.position_embedding_type not in ["rotary", "relative", "absolute"]:
            raise ValueError(
                f"Invalid positional embedding type: {self.position_embedding_type}. Options are 'rotary', 'relative', or 'absolute'."
            )
        if self.expert_capacity_type not in ["absolute", "multiplier"]:
            raise ValueError(
                f"Invalid expert capacity type: {self.expert_capacity_type}. Options are 'absolute' or 'multiplier'."
            )
        if self.expert_capacity <= 0 and self.router_type == "expert-choice":
            raise ValueError(
                f"Invalid expert capacity: {str(self.expert_capacity)}. Expert capacity must be a positive integer for the Expert Choice router."
            )
        # check base activations
        base_activations = ["gelu", "relu", "glu", "swiglu", "geglu", "reglu"]
        if self.activation not in base_activations:
            raise ValueError(
                f"Invalid FFN activation: {self.activation}. Options are 'gelu', 'relu', 'glu', 'swiglu', 'geglu', or 'reglu'."
            )
        if self.expert_activation not in base_activations:
            raise ValueError(
                f"Invalid expert activation: {self.expert_activation}. Options are 'gelu', 'relu', 'glu', 'swiglu', 'geglu', or 'reglu'."
            )
        # check head activations
        head_activations = ["tanh", "relu", "gelu"]
        if self.classifier_activation not in head_activations:
            raise ValueError(
                f"Invalid classifier activation: {self.classifier_activation}. Options are 'tanh', 'relu', or 'gelu'."
            )
        if self.mlm_activation not in head_activations:
            raise ValueError(
                f"Invalid mlm activation: {self.mlm_activation}. Options are 'tanh', 'relu', or 'gelu'."
            )
        # check classifier params
        if (
            self.attention_classifier == False
            and self.output_classifier_attentions == True
        ):
            raise ValueError(
                "Invalid classifier configuration. Cannot output classifier attentions when attention_classifier is False."
            )

    def _standardize_router_type(self, router_type: str) -> str:
        if router_type.lower() in ["topk", "top-k", "top_k"]:
            return "top-k"
        elif router_type.lower() in ["topp", "top-p", "top_p"]:
            return "top-p"
        elif router_type.lower() in ["expert choice", "expert-choice", "expert_choice"]:
            return "expert-choice"
        else:
            raise ValueError(
                f"Invalid router type: {router_type}. Options are 'top-k', 'top-p', or 'expert-choice'."
            )

    def _get_expert_sizes(
        self, expert_intermediate_size: Union[int, List[int]]
    ) -> List:
        num_experts = self.num_experts - self.num_shared_experts

        # default to intermediate size
        if expert_intermediate_size is None:
            expert_intermediate_size = self.intermediate_size

        # homogeneous experts
        if isinstance(expert_intermediate_size, int):
            self.homogeneous_experts = True
            return [expert_intermediate_size] * num_experts

        # heterogeneous experts
        elif isinstance(expert_intermediate_size, List):
            if len(expert_intermediate_size) != num_experts:
                raise ValueError(
                    f"Length of expert_intermediate_size ({len(expert_intermediate_size)}) "
                    f"must match number of non-shared experts ({num_experts})"
                )
            if not all(isinstance(x, int) for x in expert_intermediate_size):
                raise ValueError(
                    "All elements of expert_intermediate_size must be integers."
                )
            if (self.router_type == "expert-choice") and (
                len(set(expert_intermediate_size)) > 1
            ):
                raise ValueError(
                    "Heterogeneous expert sizes are not compatible with the expert choice router."
                )
            self.homogeneous_experts = False
            return expert_intermediate_size

        raise TypeError(
            "Invalid expert_intermediate_size. Must be an int, list of ints, or None."
        )

    def _get_shared_expert_size(
        self,
        shared_expert_intermediate_size: int,
        expert_intermediate_size: Union[int, List[int]],
    ) -> int:
        # return directly if provided
        if shared_expert_intermediate_size is not None:
            return int(shared_expert_intermediate_size)
        # use expert intermediate size if it was provided as an int
        if isinstance(expert_intermediate_size, int):
            return expert_intermediate_size
        # otherwise use intermediate size
        return self.intermediate_size

    def _get_classifier_activation(
        self, activation: Optional[str], use_attention: bool
    ) -> str:
        if activation is None:
            return "relu" if use_attention else "tanh"
        return activation.lower()
