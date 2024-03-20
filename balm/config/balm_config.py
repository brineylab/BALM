from typing import Optional

from transformers import PretrainedConfig


class BalmConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`SwitchTransformersModel`]. It is used to
    instantiate a SwitchTransformers model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    SwitchTransformers [google/switch-base-8](https://huggingface.co/google/switch-base-8) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Parameters:
    -----------
    vocab_size: int, optional (default=None)
        Vocabulary size of the SwitchTransformers model. Defines the number of different tokens that can be represented
        by the `inputs_ids` passed when calling [`SwitchTransformersModel`].

    hidden_size: int, optional (default=1024)
        Size of the encoder layers and the pooler layer.

    intermediate_size: int, optional (default=4096)
        Size of the intermediate feed forward layer.

    num_hidden_layers: int, optional (default=24)
        Number of dense hidden layers in the Transformer encoder layer.

    num_attention_heads: int, optional (default=16)
        Number of attention heads for each attention layer in the Transformer encoder.

    max_position_embeddings: int, optional (default=320)
        Maximum sequence length that this model might ever be used with. For paired Ab sequences, we typically
        use 320 (the defailt) and for unpaired Ab sequences we typically use 180.

    position_embedding_type: str, optional (default="absolute")
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
        For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).









    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the SwitchTransformers model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`SwitchTransformersModel`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `SwitchTransformersBlock`.
        expert_capacity (`int`, *optional*, defaults to 64):
            Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
            Transformer.
        num_layers (`int`, *optional*, defaults to 12):
            Number of dense hidden layers in the Transformer encoder layer.
        num_sparse_encoder_layers (`int`, *optional*, defaults to 6):
            Number of sparse (MoE) dense hidden layers in the Transformer encoder layer.
        num_decoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_sparse_decoder_layers (`int`, *optional*, defaults to 12):
            Number of sparse (MoE) dense hidden layers in the Transformer decoder layer.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_experts (`int`, *optional*, defaults to 8):
            Number of experts for each SwitchTransformer layer.
        router_type (`str`, *optional*, defaults to `"tokens_masked"`):
            Router type - choose between `"tokens_masked", `"tokens_scatter"` and `"experts_masked"`.
        router_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the router.
        router_jitter_noise (`float`, *optional*, defaults to 0.1):
            Amount of noise to add to the router.
        router_dtype (`str`, *optional*, default to `"float32"`):
            The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
            *selective precision* discussion in [the paper](https://arxiv.org/abs/2101.03961).
        router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
            Whether to ignore padding tokens when routing.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        router_z_loss_coef (`float`, *optional*, defaults to 0.001):
            The z loss factor for the total loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. SwitchTransformersv1.1
            uses the `"gated-gelu"` feed forward projection. Original SwitchTransformers uses `"relu"`.
        add_router_probs (`bool`, *optional*, defaults to `False`):
            Whether to output router probabilities to compute router auxiliary loss.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
    """

    model_type = "BALM_MoE"
    keys_to_ignore_at_inference = ["past_key_values"]
    # attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        max_position_embeddings: int = 320,
        position_embedding_type: str = "absolute",
        dropout_prob: float = 0.1,
        hidden_dropout_prob: Optional[float] = None,
        attention_probs_dropout_prob: Optional[float] = None,
        embedding_dropout_prob: Optional[float] = None,
        layer_norm_eps: float = 1e-12,
        initializer_factor: float = 1.0,
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = hidden_size
        self.ffn_embed_dim = intermediate_size
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        position_embedding_type = position_embedding_type.lower()
        if position_embedding_type not in [
            "absolute",
            "relative_key",
            "relative_key_query",
        ]:
            raise ValueError(
                f"`position_embedding_type` must be one of 'absolute', 'relative_key' or 'relative_key_query', got {position_embedding_type}"
            )
        self.position_embedding_type = position_embedding_type

        self.num_experts = num_experts
        if expert_capacity is None:
            expert_capacity = (
                max_position_embeddings
                / num_attention_heads
                * expert_capacity_multiplier
            )
        self.expert_capacity = int(expert_capacity)
        self.expert_activation = expert_activation

        self.router_bias = router_bias
        self.router_jitter = router_jitter
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(
                f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}"
            )
        self.router_dtype = router_dtype
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.add_router_probs = add_router_probs
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef

        self.hidden_dropout_rate = (
            hidden_dropout_prob if hidden_dropout_prob is not None else dropout_prob
        )
        self.attn_dropout_rate = (
            attention_probs_dropout_prob
            if attention_probs_dropout_prob is not None
            else dropout_prob
        )
        self.embedding_dropout_rate = (
            embedding_dropout_prob
            if embedding_dropout_prob is not None
            else dropout_prob
        )
        self.layer_norm_eps = layer_norm_eps
        self.initializer_factor = initializer_factor

        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            **kwargs,
        )