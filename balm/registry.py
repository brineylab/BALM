# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""
Setup for BALM and BALM-MoE models and tokenizers to use Hugging Face's Auto classes.
See documentation: https://huggingface.co/docs/transformers/en/custom_models#autoclass
"""

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification
)

from .models.balm import (
    BalmModel,
    BalmForMaskedLM,
    BalmForSequenceClassification
)
from .config.balm_config import BalmConfig
from .models.balm_moe import (
    BalmMoEModel,
    BalmMoEForMaskedLM,
    BalmMoEForSequenceClassification
)
from .config.balm_moe_config import BalmMoEConfig
from .tokenizer import BalmTokenizer

# tokenizer
AutoTokenizer.register("balm", fast_tokenizer_class=BalmTokenizer)
AutoTokenizer.register("balm_moe", fast_tokenizer_class=BalmTokenizer)

# BALM
AutoConfig.register("balm", BalmConfig)
AutoModel.register(BalmConfig, BalmModel)
AutoModelForMaskedLM.register(BalmConfig, BalmForMaskedLM)
AutoModelForSequenceClassification.register(BalmConfig, BalmForSequenceClassification)

# BALM MoE
AutoConfig.register("balm_moe", BalmMoEConfig)
AutoModel.register(BalmMoEConfig, BalmMoEModel)
AutoModelForMaskedLM.register(BalmMoEConfig, BalmMoEForMaskedLM)
AutoModelForSequenceClassification.register(BalmMoEConfig, BalmMoEForSequenceClassification)
