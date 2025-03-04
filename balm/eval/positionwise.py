# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import math
from typing import Optional

import torch

from ..model_utils import wrap_model


def positionwise_perplexity(
    model, dataloader, criterion, device: Optional[torch.device] = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wrap_model(model, device)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"]
            targets = batch["targets"]
            outputs = model(**inputs)
            loss = criterion(outputs.logits, targets)
            total_loss += loss.item()
            total_samples += inputs["input_ids"].size(0)
    return math.exp(total_loss / total_samples)
