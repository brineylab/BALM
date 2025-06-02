# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

# Inspired by: https://github.com/naba89/custom_hf_trainer/

from typing import Dict
import warnings

import torch
import transformers
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    is_torch_xla_available,
    EvalPrediction,
)
from packaging.version import Version

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

__all__ = ["MoETrainer", "compute_metrics_MoE"]


# MoETrainer subclasses internal transformers methods
# therefore performance is only guaranteed in certain versions
MIN_TRANSFORMERS_VERSION = Version("4.50.0")
MAX_TRANSFORMERS_VERSION = Version("4.51.0")


class AddExtraLossesToTrainerState(TrainerCallback):
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        control.extra_losses = {}
        return control


class MoETrainer(Trainer):
    """
    Custom Trainer to support logging MoE loss components during
    training and evaluation.
    """

    def __init__(self, **kwargs):
        current_version = Version(transformers.__version__)
        if not (
            MIN_TRANSFORMERS_VERSION <= current_version <= MAX_TRANSFORMERS_VERSION
        ):
            warnings.warn(
                f"MoETrainer has only been validated with transformers versions between "
                f"{MIN_TRANSFORMERS_VERSION} and {MAX_TRANSFORMERS_VERSION}. "
                f"Detected transformers=={current_version}. "
                "Compatibility issues may occur. For evaluation-only logging of MoE losses, consider "
                "using the compute_metrics_MoE function, which works across transformers versions."
            )
        
        # try importing SaveStrategy here
        # this may not work if the transformers version is wrong
        from transformers.trainer_utils import SaveStrategy

        super().__init__(**kwargs)

        # add callback for logging extra train losses
        self.add_callback(AddExtraLossesToTrainerState())

        # provide compute metrics (if not provided) for logging extra eval losses
        if self.compute_metrics is None:
            self.compute_metrics = compute_metrics_MoE

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if hasattr(self.control, "extra_losses") and model.training:

            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

            if not isinstance(outputs, dict):
                raise ValueError(
                    "The model output should be a dictionary or ModelOutput and not a tuple or list."
                )

            # extract extra losses from outputs
            if hasattr(outputs, "moe_losses"):
                for k, v in outputs.moe_losses.items():
                    # add key if it doesn't exist
                    if k not in self.control.extra_losses:
                        self.control.extra_losses[k] = 0.0

                    if self.args.n_gpu > 1:
                        v = v.mean()
                    self.control.extra_losses[k] += (
                        v.detach() / self.args.gradient_accumulation_steps
                    )

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            # added to log extra losses
            if hasattr(self.control, "extra_losses"):
                for k, v in self.control.extra_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    # reset the loss
                    self.control.extra_losses[k] -= self.control.extra_losses[k]

                    logs[k] = logs[k] / (
                        self.state.global_step - self._globalstep_last_logged
                    )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


def compute_metrics_MoE(eval_pred: EvalPrediction):
    predictions, _ = eval_pred

    metrics = {}
    if isinstance(predictions, tuple):
        # moe_losses dict is always the final element
        final = predictions[-1]
        if isinstance(final, dict):
            for k, v in final.items():
                metrics[k] = v.mean() if hasattr(v, "mean") else float(v)

    return metrics
