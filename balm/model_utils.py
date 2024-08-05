# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Union

import torch

__all__ = ["wrap_model", "DTYPES"]


def wrap_model(
    model: torch.nn.Module, device: Union[torch.device, str]
) -> torch.nn.Module:
    """
    Wraps a model with a device and optionally a DataParallel wrapper.

    Parameters
    ----------
    model : torch.nn.Module
        The model to wrap.
    device : torch.device or str
        The device to wrap the model on.

    Returns
    -------
    torch.nn.Module
        The wrapped model.
    """
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}
