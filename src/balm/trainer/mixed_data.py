# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

# Modified from our curriculum paper: https://github.com/brineylab/curriculum-paper

from accelerate import Accelerator
from accelerate.state import AcceleratorState
import torch
from transformers import TrainerCallback
import wandb

from ..datasets import MixedProbDataset

__all__ = ["MixedDatasetCallback"]


class MixedDatasetCallback(TrainerCallback):
    def __init__(self, dataset: MixedProbDataset):
        super().__init__()
        self.dataset = dataset
        self.accelerator = None

        # for tracking eval losses
        self.eval_paired = None
        self.eval_unpaired = None

    def on_train_begin(self, args, state, control, **kwargs):

        # set accelerator after it's initialized by trainer
        if self.accelerator is None:
            self.accelerator = Accelerator()

        # set dataset generator
        self.dataset.set_generator(accelerator=self.accelerator)

        # log initial values
        if state.is_local_process_zero and wandb.run:
            wandb.log(
                {
                    "train/unpaired_epoch": 0,
                    "train/paired_epoch": 0,
                    "train/unpaired_probability": self.dataset.current_prob,
                },
                step=0,
            )

    # update probability with current train step
    def on_step_begin(self, args, state, control, **kwargs):
        self.dataset.set_current_step(state.global_step)

    # add extra train and eval losses
    def on_log(self, args, state, control, logs=None, **kwargs):

        def calculate_epoch(count, dataset):
            if dataset:
                count_tensor = torch.tensor(
                    [count], dtype=torch.int64, device=self.accelerator.device
                )
                reduced = self.accelerator.reduce(count_tensor, reduction="sum").item()
                return round(reduced / len(dataset), 4)
            return 0

        added_logs = {}

        # eval logs
        if any(k.startswith("eval_") for k in logs):
            self.eval_unpaired = logs.get("eval_unpaired_loss", self.eval_unpaired)
            self.eval_paired = logs.get("eval_paired_loss", self.eval_paired)

            # check if both paired and unpaired losses have been calculated
            if self.eval_unpaired is not None and self.eval_paired is not None:
                # calculate average & weighted losses
                avg_loss = (self.eval_unpaired + self.eval_paired) / 2
                weighted_loss = (
                    self.eval_unpaired * self.dataset.current_prob
                    + self.eval_paired * (1 - self.dataset.current_prob)
                )

                # reset before next eval step
                self.eval_unpaired, self.eval_paired = None, None

                # update logs
                added_logs.update(
                    {
                        "eval/eval_average_loss": avg_loss,
                        "eval/eval_weighted_loss": weighted_loss,
                    }
                )
        # train logs
        else:
            # calculate epochs
            unpaired_epoch = calculate_epoch(
                self.dataset.unpaired_count, self.dataset.unpaired_data
            )
            paired_epoch = calculate_epoch(
                self.dataset.paired_count, self.dataset.paired_data
            )

            # update logs
            added_logs.update(
                {
                    "train/unpaired_epoch": unpaired_epoch,
                    "train/paired_epoch": paired_epoch,
                    "train/unpaired_probability": self.dataset.current_prob,
                }
            )

        # update logs
        if state.is_local_process_zero and added_logs:
            logs.update(added_logs)
            if wandb.run is not None:
                wandb.log({**added_logs, "train/global_step": state.global_step})
