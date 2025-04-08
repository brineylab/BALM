# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["get_activation_fn"]


def get_activation_fn(
    activation: Union[str, nn.Module],
    # extra params for GLU variants
    input_dim: int | None = None,
    output_dim: int | None = None,
    bias: bool | None = None
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
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "gelu":
            return GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation in GLU_variants.keys():
            return GLU(
                in_dim=input_dim, 
                out_dim=output_dim, 
                activation=activation, 
                bias=bias
            )
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


GLU_variants = {
    'glu': F.sigmoid,
    'swiglu': F.silu,
    'geglu': F.gelu,
    'reglu': F.relu,
}


class GLU(nn.Module):
    """
    Activation function for GLU variants.

    Parameters
    ----------
    in_dim: int
        Input dimension.
    out_dim: int
        Output dimension.
    activation: str
        Type of GLU variant.
    bias: bool
        Whether to use bias in linear layers.

    """
    def __init__(self, in_dim: int, out_dim: int, activation: str, bias: bool):
        super().__init__()
        self.value_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.gate_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation_fn = GLU_variants[activation]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_linear(x)
        gate = self.gate_linear(x)
        return value * self.activation_fn(gate)

# class GLU(nn.Module):
#     """
#     Activation function for GLU variants.

#     Parameters
#     ----------
#     in_dim: int
#         Input dimension.
#     out_dim: int
#         Output dimension.
#     activation: str
#         Type of GLU variant.
#     bias: bool
#         Whether to use bias in linear layers.

#     """
#     def __init__(self, in_dim: int, out_dim: int, activation: str, bias: bool):
#         super().__init__()
#         self.wi = nn.Linear(in_dim, (out_dim*2), bias=bias)
#         self.activation_fn = GLU_variants[activation]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.wi(x)
#         value, gate = x.chunk(2, dim=-1)
#         return value * self.activation_fn(gate)
