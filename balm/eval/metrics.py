# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import evaluate
import numpy as np

# import sklearn as skl
import torch
from sklearn.metrics import (
    # accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "ComputeMetrics",
    # "ComputeMetricsForMaskedLM",
    # "ComputeMetricsForSequenceClassification",
]


accuracy_score = evaluate.load("accuracy")


class ComputeMetrics:
    def __init__(self, positive_label: int = 1):
        self.positive_label = positive_label
        self.head_loss = None
        self.z_loss = None
        self.aux_loss = None

    def __call__(self, eval_preds):
        # process eval_preds
        self.logits, self.labels = eval_preds
        if isinstance(self.logits, tuple):
            self.logits = self.logits[0]

        # compute probabilities and predictions
        self.probabilities = (
            torch.softmax(torch.from_numpy(self.logits), dim=1).detach().numpy()[:, -1]
        )
        self.predictions = np.argmax(self.logits, axis=1)

        # build outputs
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "auroc": self.auroc(),
            "aupr": self.aupr(),
            "f1": self.f1(),
            "mcc": self.mcc(),
        }

    def accuracy(self):
        accuracy = accuracy_score.compute(
            predictions=self.predictions, references=self.labels
        )
        return accuracy["accuracy"]

    def precision(self):
        precision = precision_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )
        return precision

    def recall(self):
        recall = recall_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )
        return recall

    def f1(self):
        f1 = f1_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )
        return f1

    def auroc(self):
        auroc = roc_auc_score(y_true=self.labels, y_score=self.probabilities)
        return auroc

    def aupr(self):
        aupr = average_precision_score(
            y_true=self.labels,
            y_score=self.probabilities,
            pos_label=self.positive_label,
        )
        return aupr

    def mcc(self):
        mcc = matthews_corrcoef(y_true=self.labels, y_pred=self.predictions)
        return mcc


# class ComputeMetricsForMaskedLM(ComputeMetricsBase):
#     head_loss_name: str = "lm_loss"

#     def __init__(self, positive_label: int = 1):
#         super().__init__(positive_label=positive_label)


# class ComputeMetricsForSequenceClassification(ComputeMetricsBase):
#     head_loss_name: str = "classifier_loss"

#     def __init__(self, positive_label: int = 1):
#         super().__init__(positive_label=positive_label)
