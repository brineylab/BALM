# Copyright (c) 2024 brineylab @ scripps
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
        activation: str = "swiglu",
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        hidden_dropout: Optional[float] = None,
        max_position_embeddings: int = 320,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "rotary",
        # pre_norm: bool = True,
        token_dropout: float = 0.0,
        mask_token_id: int = 31,
        pad_token_id: int = 1,
        type_vocab_size: int = 2,
        # classification
        mlm_activation: str = "gelu",
        classifier_activation: str = "tanh",
        num_labels: int = 2,  # sequence/token-level classification
        num_choices: int = 4,  # multiple choice classification
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
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.position_embedding_type = position_embedding_type.lower()
        # self.pre_norm = bool(pre_norm)
        self.token_dropout = float(token_dropout)
        self.type_vocab_size = int(type_vocab_size)
        self.use_cache = bool(use_cache)

        # classification
        self.num_labels = int(num_labels)
        self.num_choices = int(num_choices)
        self.classifier_activation = classifier_activation.lower()
        self.mlm_activation = mlm_activation.lower()

        # outputs
        self.return_dict = bool(return_dict)
        self.output_attentions = bool(output_attentions)
        self.output_hidden_states = bool(output_hidden_states)

        # validate params
        if self.position_embedding_type not in ["rotary", "relative", "absolute"]:
            raise ValueError(
                f"Invalid positional embedding type: {self.position_embedding_type}. Options are 'rotary', 'relative', or 'absolute'."
            )
        if self.activation not in ["swiglu", "relu", "gelu", "glu", "reglu", "geglu"]:
            raise ValueError(
                f"Invalid FFN activation: {self.activation}. Options are 'swiglu', 'relu', 'gelu', 'glu', 'reglu', or 'geglu'."
            )
        if self.classifier_activation not in [
            "swiglu",
            "relu",
            "gelu",
            "tanh",
            "reglu",
            "geglu",
        ]:
            raise ValueError(
                f"Invalid classifier activation: {self.classifier_activation}. Options are 'swiglu', 'relu', 'gelu', 'tanh', 'reglu', or 'geglu'."
            )
