# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import json
import os
import re
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

WEIGHTS_NAME = "model.pt"
SAFE_WEIGHTS_NAME = "model.safetensors"

__all__ = ["BalmPreTrainedModel", "FreezeBaseModelMixin", "ParameterCountMixin"]


class BalmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    supports_gradient_checkpointing = True

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
        if hasattr(self, "router_z_loss_coef"):
            self.router_z_loss_coef = 0.0
        if hasattr(self, "router_aux_loss_coef"):
            self.router_aux_loss_coef = 0.0


class ParameterCountMixin:
    def count_parameters(
        self,
        only_trainable: bool = True,
        exclude_embeddings: bool = False,
        human_readable: bool = False,
    ) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Parameters
        ----------
        only_trainable : bool, optional, defaults to `False`
            Whether or not to return only the number of trainable parameters

        exclude_embeddings : bool, optional, defaults to `False`
            Whether or not to return only the number of non-embeddings parameters

        human_readable : bool, optional, defaults to `False`
            Whether or not to return the number of parameters in a human-readable format

        Returns
        -------
        int or str
            The number of parameters. If `human_readable` is `True`, the number of parameters
            will be returned as a string in a human-readable format.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter
                for name, parameter in self.named_parameters()
                if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                total_numel.append(param.numel())
        total_num_params = sum(total_numel)

        if human_readable:
            if total_num_params < 1e3:
                return total_num_params
            elif total_num_params < 1e6:
                return f"{total_num_params / 1e3:.2f}K"
            elif total_num_params < 1e9:
                return f"{total_num_params / 1e6:.2f}M"
            elif total_num_params < 1e12:
                return f"{total_num_params / 1e9:.2f}B"
            else:
                return f"{total_num_params / 1e12:.2f}T"
        else:
            return total_num_params
