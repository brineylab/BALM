# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

from transformers import PretrainedConfig


class BalmConfig(PretrainedConfig):
    """
    Configuration for Balm models.

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

    activation : str, default="swiglu"
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
        Whether to use a bias for FFN layers.

    max_position_embeddings : int, default=256
        The maximum position embeddings.

    initializer_range : float, default=0.02
        The initializer range for the model.

    layer_norm_eps : float, default=1e-5
        The epsilon for layer normalization.

    position_embedding_type : str, default="rotary"
        The type of position embeddings to use.
        Options are "rotary" or "absolute".

    mask_token_id : int, default=31
        The mask token id.

    pad_token_id : int, default=1
        The pad token id.

    mlm_activation: str, default="gelu"
        The activation function to use for the LM head.
         
    attention_classifier: bool, default=False
        Whether to add attention to classification head.
    
    classifier_attention_heads: int, default=None
        Number of attention heads in the classifier. 
        If not provided, defaults to `num_attention_heads`.
        Only used if `attention_classifier` is True.

    classifier_activation: str, default="tanh"
        The activation function to use for the classifier.

    classifier_freeze_base: bool, default=True
        Whether to freeze the base weights of classification model. 
    
    num_labels : int, default=2
        The number of labels for the classification head (sequence or token classification).
    
    output_classifier_attentions : bool, default=False
        Whether to output the classifier attention. 
        Only used if `attention_classifier` is True.

    output_attentions : bool, default=False
        Whether to output the attentions.

        .. warning::
            If `output_attentions` is True, torch can't use optimized SDPA.
            See `here`_ for more details.

    output_hidden_states : bool, default=False
        Whether to output the hidden states.

    return_dict: bool, default = True
        Whether to return a dictionary of outputs (returns a tuple if False).

    use_cache : bool, default=True
        Whether to use the cache.

    **kwargs : dict, optional
        Additional keyword arguments are passed to the parent class (`transformers.PretrainedConfig`).

    Raises
    ------
    ValueError
        If the positional embedding type is not valid.

    ValueError
        If the FFN, mlm, or classifier activation functions are not valid.
    
    ValueError
        If the classifier config is not valid.

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
        activation: str = "swiglu",
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        hidden_dropout: Optional[float] = None,
        ffn_bias: bool = True,
        max_position_embeddings: int = 256,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        position_embedding_type: str = "rotary",
        mask_token_id: int = 31,
        pad_token_id: int = 1,
        # mlm
        mlm_activation: str = "gelu",
        # classification
        attention_classifier: bool = False,
        classifier_attention_heads: Optional[int] = None,
        classifier_activation: str = "tanh",
        classifier_freeze_base: bool = True,
        num_labels: int = 2,  # sequence/token-level classification
        output_classifier_attentions: bool = False,
        # outputs
        output_attentions: bool = False,
        output_hidden_states: bool = False,
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
        
        # mlm
        self.mlm_activation = mlm_activation.lower()

        # classification
        self.attention_classifier = bool(attention_classifier)
        self.classifier_attention_heads = int(
            classifier_attention_heads if classifier_attention_heads is not None else num_attention_heads
        )
        self.classifier_activation = classifier_activation.lower()
        self.classifier_freeze_base = bool(classifier_freeze_base)
        self.num_labels = int(num_labels)
        self.output_classifier_attentions = bool(output_classifier_attentions)

        # outputs
        self.output_attentions = bool(output_attentions)
        self.output_hidden_states = bool(output_hidden_states)
        self.return_dict = bool(return_dict)

        # 🤗 integration
        self.use_cache = bool(use_cache)

        # validate params
        if self.position_embedding_type not in ["rotary", "relative", "absolute"]:
            raise ValueError(
                f"Invalid positional embedding type: {self.position_embedding_type}. Options are 'rotary', 'relative', or 'absolute'."
            ) 
        # check base activations
        base_activations = ["gelu", "relu", "glu", "swiglu", "geglu", "reglu"]
        if self.activation not in base_activations:
            raise ValueError(
                f"Invalid FFN activation: {self.activation}. Options are 'gelu', 'relu', 'glu', 'swiglu', 'geglu', or 'reglu'."
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
        if self.attention_classifier == False and self.output_classifier_attentions == True:
            raise ValueError(
                "Invalid classifier configuration. Cannot output classifier attentions when attention_classifier is False."
            )
