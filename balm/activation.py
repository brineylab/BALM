# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwiGLU", "get_activation_fn"]


def get_activation_fn(activation: Union[str, nn.Module]) -> nn.Module:
    """
    Get an activation function from a string or a PyTorch module.

    Parameters
    ----------
    activation: Union[str, nn.Module]
        The activation function to get. If a string, it must be one of "tanh", "gelu", "relu", or "swiglu".
        If a module, it must be a subclass of `torch.nn.Module`, and the module will be returned as is.

    Returns
    -------
    nn.Module
        The activation function.

    Raises
    ------
    ValueError
        If the activation function is not supported.

    """
    if isinstance(activation, str):
        activation = activation.lower()
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swiglu":
            return SwiGLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    raise ValueError(
        f"Activation must be a string or a PyTorch module, got {type(activation)}"
    )


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 + F.silu(x2)
