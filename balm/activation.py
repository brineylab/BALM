# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwiGLU", "GeGLU", "ReGLU", "get_activation_fn"]


def get_activation_fn(
    activation: Union[str, nn.Module],
    dim: int | None = None,
) -> nn.Module:
    """
    Get an activation function from a string or a PyTorch module.

    Parameters
    ----------
    activation: Union[str, nn.Module]
        The activation function to get. If a string, it must be one of "tanh", "gelu", "relu", "glu",
        "swiglu", "geglu", or "reglu". If a module, it must be a subclass of `torch.nn.Module`,
        and the module will be returned as is.

    .. warning::
        SwiGLU will return a tensor with half the dimension of the input tensor.

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
            return GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "glu":
            return nn.GLU()
        elif activation == "swiglu":
            return SwiGLU()
        elif activation == "geglu":
            return GeGLU()
        elif activation == "reglu":
            return ReGLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    raise ValueError(
        f"Activation must be a string or a PyTorch module, got {type(activation)}"
    )


class GELU(nn.Module):
    """
    GELU activation function from original ESM repo. 
    Using F.gelu yields subtly wrong results.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    Parameters
    ----------
    dim: int | None, default=None
        Model dimension. 
        If provided, the input tensor will be separately processed by two linear layers.
        If not provided, the input tensor will be chunked into two tensors, and the 
        resulting two tensors will be used to compute the SwiGLU activation. This results
        in the output dimension being half of the input dimension.

    Returns
    -------
    torch.Tensor
        The SwiGLU activation of the input tensor.

    """

    def __init__(self, dim: int = None):
        super().__init__()

        self.dim = dim
        if dim is not None:
            self.value_linear = nn.Linear(dim, dim)
            self.gate_linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim:
            value = self.value_linear(x)
            gate = self.gate_linear(x)
        else:
            value, gate = x.chunk(2, dim=-1)
        
        return value * F.silu(gate)


class GeGLU(nn.Module):
    """
    GeGLU activation function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.gelu(x2)


class ReGLU(nn.Module):
    """
    ReGLU activation function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.relu(x2)
