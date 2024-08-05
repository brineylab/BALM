# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwiGLU"]


class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.sigmoid(x1) * x2
