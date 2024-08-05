# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Callable, Optional, Tuple

from balm.data import Dataset
from balm.models.base import BalmBase


def binary_classification(
    model: BalmBase, dataset: Dataset, compute_metrics: Optional[Callable] = None
) -> Tuple[float, float, float, float]:
    """
    Evaluate a binary classification model.

    Parameters
    ----------
    model : BalmBase
        The model to evaluate.
    dataset : Dataset
        The dataset to evaluate on.
    compute_metrics : Optional[Callable]
        A function to compute metrics.

    Returns
    -------
    Tuple[float, float, float, float]
        The precision, recall, F1, and accuracy of the model.
    """
    pass
