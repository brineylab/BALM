from transformers import Trainer, TrainerCallback

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
        z_loss = outputs.z_loss.mean()
        aux_loss = outputs.aux_loss.mean() 
        lm_loss = outputs.lm_loss.mean()

        self.state.custom_log_cache = {
            "z_loss": z_loss.item(),
            "aux_loss": aux_loss.item(),
            "lm_loss": lm_loss.item(),
        }

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs):
        if self.model.training:
            logs.update(self.state.custom_log_cache) # merge trainer's logs with custom losses
        super().log(logs)
