# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

# Inspired by: https://github.com/naba89/custom_hf_trainer/

from typing import Dict, List, Optional

import torch
from transformers import Trainer

from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    is_torch_xla_available,
    EvalPrediction,
)
from transformers.trainer_utils import SaveStrategy

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

__all__ = ["MoETrainer"]


loss_mapping = {
    "topk": ["lm_loss", "aux_loss", "z_loss"],
    "topk-penalty": ["lm_loss", "penalty_loss"],
    "topk-aux": ["lm_loss", "aux_loss"],
    "expert choice": ["lm_loss", "z_loss"],
}


class AddExtraLossesToTrainerState(TrainerCallback):
    def __init__(self, extra_losses: List[str]):
        self.extra_losses = extra_losses

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        control.extra_losses = {
            k: torch.tensor(0.0).to(args.device) for k in self.extra_losses
        }
        return control


class MoETrainer(Trainer):
    """
    Custom Trainer to support logging MoE loss components during
    training and evaluation.
    """

    def __init__(self, extra_losses: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)

        # extra_losses based on router type (if not provided)
        self.router_type = self.model.config.router_type
        if self.model.config.homogeneous_experts:
            router_str = self.router_type
        else:
            penalty = self.model.config.router_use_penalty_loss
            router_str = f"{self.router_type}{'-penalty' if penalty else '-aux'}"
        
        extra_losses = (
            loss_mapping[router_str] if extra_losses is None else extra_losses
        )

        # add callback for logging extra train losses
        self.add_callback(AddExtraLossesToTrainerState(extra_losses))

        # provide compute metrics (if not provided) for logging extra eval losses
        if self.compute_metrics is None:
            self.compute_metrics = self._compute_metrics

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
            for k, v in outputs.items():
                if k in self.control.extra_losses:
                    if v is not None:
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

    def _compute_metrics(self, eval_pred: EvalPrediction):
        predictions, labels = eval_pred

        metrics = {}
        if isinstance(predictions, tuple):
            # top k
            if self.router_type == "topk" and len(predictions) == 4:
                _, z_loss, aux_loss, lm_loss = predictions
                metrics["lm_loss"] = lm_loss.mean()
                metrics["z_loss"] = z_loss.mean()
                metrics["aux_loss"] = aux_loss.mean()
            elif (
                self.router_type == "topk" and len(predictions) == 3
            ):  # homogeneous experts
                _, penalty_loss, lm_loss = predictions
                metrics["lm_loss"] = lm_loss.mean()
                metrics["penalty_loss"] = penalty_loss.mean()
            # expert choice
            elif self.router_type == "expert choice" and len(predictions) == 3:
                _, z_loss, lm_loss = predictions
                metrics["lm_loss"] = lm_loss.mean()
                metrics["z_loss"] = z_loss.mean()

        return metrics
