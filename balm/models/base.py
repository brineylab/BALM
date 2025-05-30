# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch.nn as nn
from transformers import PreTrainedModel

from ..modules import SparseFFN
from ..loss import ROUTER_LOSSES

WEIGHTS_NAME = "model.pt"
SAFE_WEIGHTS_NAME = "model.safetensors"

__all__ = ["BalmPreTrainedModel", "FreezeBaseModelMixin", "ParameterCountMixin"]


class BalmPreTrainedModel(PreTrainedModel):
    """
    Handles weight initialization and provides an interface for downloading
    and loading pretrained models. Inherits from `PreTrainedModel` in
    HuggingFace Transformers.
    """

    supports_gradient_checkpointing = True

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # router loss coefficients
        # extract values as a dict, so that we can zero them out in freeze_base_model()
        # and access them when calculating the loss
        if self.base_model_prefix == "balm_moe":
            self.router_loss_coeffs = {
                k: getattr(config, f"router_{k}_coef")
                for k in ROUTER_LOSSES
                if hasattr(config, f"router_{k}_coef")
            }

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FreezeBaseModelMixin:
    """
    Mixin class to freeze the base model's parameters during training.

    This is utilized in the SequenceClassification models, to train only the
    classifier head while keeping the base model frozen.
    """

    def freeze_base_model(self, base_model: Optional[str] = None):
        if base_model is None:
            if not hasattr(self.__class__, "base_model_prefix"):
                raise ValueError(
                    f"{self.__class__.__name__} does not have a default base model prefix, so you need to provide `base_model`."
                )
            base_model = getattr(self, self.__class__.base_model_prefix)
        else:
            if not hasattr(self, base_model):
                raise ValueError(
                    f"This model instance does not have the supplied base model ({base_model})."
                )
            base_model = getattr(self, base_model)
        for param in base_model.parameters():
            param.requires_grad = False

        # zero out router loss coefficients (MoE models only)
        if hasattr(self, "router_loss_coeffs"):
            for k in self.router_loss_coeffs:
                self.router_loss_coeffs[k] = 0.0


class ParameterCountMixin:
    """
    Mixin class to count model parameters.

    Supports counting:
      - All parameters
      - Only trainable parameters
      - Only active parameters (for MoE models)
    """

    def count_parameters(
        self,
        only_trainable: bool = True,
        only_active: bool = False,
        num_tokens: int = 0,
        exclude_embeddings: bool = False,
        human_readable: bool = False,
    ) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Parameters
        ----------
        only_trainable : bool, default=`True`
            Whether or not to return only the number of trainable parameters
        only_active: bool, default=`False`
            Whether or not to return only the number of active parameters.
            Raises ValueError if no MoE layers are found.
        num_tokens: int, default=0
            The number of tokens processed per batch.
        exclude_embeddings : bool, default=`False`
            Whether or not to return only the number of non-embeddings parameters
        human_readable : bool, default=`False`
            Whether or not to return the number of parameters in a human-readable format

        Returns
        -------
        int or str
            The number of parameters. If `human_readable` is `True`, the number of parameters
            will be returned as a string in a human-readable format.
        """

        # check if embeddings should be excluded
        if exclude_embeddings:
            exclude_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, nn.Embedding)
            ]
        else:
            exclude_param_names = []

        # count only active parameters
        if only_active:
            # locate MoE layers
            moe_layers = [
                module for module in self.modules() if isinstance(module, SparseFFN)
            ]
            if not moe_layers:
                raise ValueError("No MoE layers were found.")

            # get the necessary config values
            num_shared_experts = self.config.num_shared_experts
            num_experts = (
                self.config.num_experts - num_shared_experts
            )  # excluding shared experts
            capacity_type = self.config.expert_capacity_type
            expert_capacity = self.config.expert_capacity

            # checks
            if capacity_type == "absolute" and num_tokens == 0:
                raise ValueError(
                    "To calculate the number of active parameters, you must specify the number of tokens per batch "
                    "when the capacity type is 'absolute'"
                )

            # calculate proportion of tokens that each expert receives
            if expert_capacity == -1:
                k = self.config.num_experts_per_tok
                prop_tokens_per_expert = k / num_experts
            elif capacity_type == "absolute":
                prop_tokens_per_expert = expert_capacity / num_tokens
            else:
                prop_tokens_per_expert = expert_capacity / num_experts

            # count dense parameters
            total_num_params = sum(
                p.numel()
                for name, p in self.named_parameters()
                if (not any(p is e for moe in moe_layers for e in moe.parameters()))
                and (name not in exclude_param_names)
                and (not only_trainable or p.requires_grad)
            )

            # count sparse parameters
            for moe in moe_layers:
                # router (fully active)
                router_params = sum(p.numel() for p in moe.router.parameters())
                total_num_params += router_params

                # shared experts, if any (fully active)
                if num_shared_experts > 0:
                    shared_expert_params = sum(
                        p.numel() for p in moe.shared_experts.parameters()
                    )
                    total_num_params += shared_expert_params

                # experts (partially active)
                total_expert_params = sum(p.numel() for p in moe.experts.parameters())
                active_expert_params = total_expert_params * prop_tokens_per_expert
                total_num_params += active_expert_params
        # count all parameters
        else:
            total_num_params = sum(
                p.numel()
                for name, p in self.named_parameters()
                if name not in exclude_param_names
                and (not only_trainable or p.requires_grad)
            )

        if human_readable:
            return self._human_readable(total_num_params)
        return total_num_params

    def _human_readable(self, total_num_params):
        units = ["T", "B", "M", "K"]
        thresholds = [1e12, 1e9, 1e6, 1e3]

        for unit, threshold in zip(units, thresholds):
            if total_num_params >= threshold:
                return f"{total_num_params / threshold:.2f}{unit}"

        return str(total_num_params)
