# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Union

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

    model_dim: int | None
        The dimension of the input tensor. Used only for SwiGLU activation.
        If provided, the input tensor will be separately processed by two linear layers.
        If not provided, the input tensor will be chunked into two tensors, and the resulting two tensors
        will be used to compute the SwiGLU activation (with the return tensor being half the size of the input tensor).

    ffn_dim: int | None
        The dimension of the Ffeed-forward network. Used only for SwiGLU activation. If not provided,
        the FFN dimension will be set to 4x the model_dim.

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
        elif activation == "glu":
            return nn.GLU()
        elif activation == "swiglu":
            return SwiGLU(dim=dim)
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


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    .. note::
        Depending on whether the dimension is provided, the implementation will be different.

        If the dimension is provided, the input tensor will separately processed by two linear layers,
        and resulting two tensors will be used to compute the SwiGLU activation, like so:

        ```python
        gate = self.gate_linear(x)
        value = self.value_linear(x)
        return value * F.silu(gate)
        ```

        If the dimension is not provided, the input tensor will be chunked into two tensors,
        and the resulting two tensors will be used to compute the SwiGLU activation. This results
        in the output dimension being half of the input dimension, like so:

        ```python
        value, gate = x.chunk(2, dim=-1)
        return value + F.silu(gate)
        ```

    Parameters
    ----------
    dim: int | None
        Model dimension. If provided, the input tensor will be separately processed by two linear layers.
        If not provided, the input tensor will be chunked into two tensors, and the resulting two tensors
        will be used to compute the SwiGLU activation (with the return tensor being half the size of the input tensor).

    Returns
    -------
    torch.Tensor
        The SwiGLU activation of the input tensor.

    """

    def __init__(self, dim: int | None = None):
        super().__init__()
        if dim is not None:
            self.gate_linear = nn.Linear(dim, dim)
            self.value_linear = nn.Linear(dim, dim)
            self.chunked_version = False
        else:
            self.chunked_version = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunked_version:
            value, gate = x.chunk(2, dim=-1)
            return value + F.silu(gate)
        else:
            gate = self.gate_linear(x)
            value = self.value_linear(x)
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
