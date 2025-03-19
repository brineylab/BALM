# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from transformers import Trainer


class LossLoggingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state.custom_log_cache = {} 

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)

        # leave loss untouched
        loss = outputs.loss

        # mean losses across batch
        loss_manual = loss.mean()
        lm_loss = outputs.lm_loss.mean()
        z_loss = outputs.z_loss.mean()
        aux_loss = outputs.aux_loss.mean() if outputs.aux_loss is not None else None

        self.state.custom_log_cache = {
            "loss_manual": loss_manual.item(),
            "lm_loss": lm_loss.item(),
            "z_loss": z_loss.item(),
        }
        if aux_loss is not None:
            self.state.custom_log_cache["aux_loss"] = aux_loss.item()

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs):
        if self.model.training:
            logs.update(self.state.custom_log_cache) # merge trainer's logs with custom losses
        super().log(logs)
