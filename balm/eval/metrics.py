# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import evaluate
import numpy as np

# import sklearn as skl
import torch
import torch.nn.functional as F
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
    # "ComputeMetricsBase",
    "ComputeMetricsForMaskedLM",
    "ComputeMetricsForSequenceClassification",
]


accuracy_score = evaluate.load("accuracy")


class ComputeMetricsBase:
    def __init__(self, positive_label: int = 1):
        self.positive_label = positive_label
        self.lm_loss = None
        self.z_loss = None
        self.aux_loss = None

    def __call__(self, eval_preds):
        raise NotImplementedError

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


class ComputeMetricsForMaskedLM(ComputeMetricsBase):
    def __init__(self, positive_label: int = 1):
        super().__init__(positive_label=positive_label)

    def __call__(self, eval_preds):
        # process eval_preds
        self.logits, self.labels = eval_preds
        if isinstance(self.logits, tuple):
            # logits is the first element of the tuple
            self.logits = self.logits[0]

            # the remaining elements may be extra losses (from MoE routers)
            if len(eval_preds[0]) >= 2:
                other_data = eval_preds[0][1:]
                batch_size = self.logits.shape[-1]  # self.logits is a numpy array
                # final element might the MaskedLM head loss
                if other_data[-1].shape == torch.Size(
                    [
                        batch_size,
                    ]
                ):
                    self.lm_loss = np.mean(other_data[-1])
                # next to last element might be z-loss
                if len(other_data) >= 2:
                    if other_data[-2].shape == (batch_size,):
                        self.z_loss = np.mean(other_data[-2])
                # third to last might be aux loss
                if len(other_data) >= 3:
                    if other_data[-3].shape == (batch_size,):
                        self.aux_loss = np.mean(other_data[-3])

        self.logits = torch.from_numpy(self.logits).detach().cpu()
        self.labels = torch.from_numpy(self.labels).detach().cpu()

        # build output
        return_dict = {
            "accuracy": self.accuracy(),
            "perplexity": self.perplexity(),
        }
        if self.lm_loss is not None:
            return_dict["mlm loss"] = self.lm_loss
        if self.z_loss is not None:
            return_dict["z loss"] = self.z_loss
        if self.aux_loss is not None:
            return_dict["aux loss"] = self.aux_loss

        return return_dict

    def accuracy(self):
        predictions = F.softmax(self.logits, dim=-1).argmax(dim=-1)
        label_mask = self.labels != -100
        correct_predictions = torch.sum((predictions == self.labels) * label_mask)
        return correct_predictions.float() / torch.sum(label_mask)

    def perplexity(self):
        logits_flat = self.logits.view(-1, self.logits.size(-1))
        labels_flat = self.labels.view(-1)
        ce_loss = F.cross_entropy(
            logits_flat, labels_flat, ignore_index=-100, reduction="mean"
        )
        perplexity = torch.exp(ce_loss)
        return perplexity.item()


class ComputeMetricsForSequenceClassification(ComputeMetricsBase):
    def __init__(self, positive_label: int = 1):
        super().__init__(positive_label=positive_label)

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
