import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import BartForConditionalGeneration

from ..scheduler import LinearWarmupLR


class R3FModule(pl.LightningModule):
    """pytorch lightning module to train BART for dialogue summarization using R3F
    https://arxiv.org/abs/2008.03156v1

    Attributes:
        model: BART model
        total_steps: total training steps for lr scheduling
        max_learning_rate: Max LR
        min_learning_rate: Min LR
        warmup_rate: warmup step rate
        model_save_dir: path to save model
        r3f_lambda: R3F parameter
    """

    def __init__(
        self,
        model: BartForConditionalGeneration,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: float,
        model_save_dir: str,
        r3f_lambda: float = 1.0,
    ):
        super().__init__()

        self.save_hyperparameters(
            {
                **model.config.to_dict(),
                "total_steps": total_steps,
                "max_learning_rate": max_learning_rate,
                "min_learning_rate": min_learning_rate,
                "r3f_lambda": r3f_lambda,
            }
        )

        self.model = model
        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_rate = warmup_rate
        self.model_save_dir = model_save_dir
        self.r3f_lambda = r3f_lambda

        self.noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=1e-5)

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
        ) / noised_logits.size(0)

    def training_step(self, batch, batch_idx):
        inputs_embeds = self.model.model.shared(batch["input_ids"])
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )

        noise = self.noise_sampler.sample(sample_shape=inputs_embeds.shape).to(inputs_embeds)
        noise_embeds = inputs_embeds.detach().clone() + noise
        noise_output = self.model(
            inputs_embeds=noise_embeds,
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])
        noise_logits = noise_output["logits"][:, :-1].reshape([labels.shape[0], -1])

        symm_kl = self._get_symm_kl(noise_logits, logits)
        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)
        loss += self.r3f_lambda * symm_kl
        accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=self.model.config.pad_token_id)

        metrics = {"loss": loss, "acc": accuracy}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])

        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)
        accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=self.model.config.pad_token_id)

        metrics = {"val_loss": loss, "val_acc": accuracy}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return metrics

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.max_learning_rate)
        scheduler = LinearWarmupLR(
            optimizer,
            int(self.total_steps * self.warmup_rate),
            self.total_steps,
            self.min_learning_rate / self.max_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "Learning Rate"},
        }

    def validation_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)

        if self.trainer.is_global_zero:
            val_losses = [output["val_loss"].mean() for output in outputs]
            val_accs = [output["val_acc"].mean() for output in outputs]

            val_loss_mean = sum(val_losses) / len(val_losses)
            val_acc_mean = sum(val_accs) / len(val_accs)

            self.model.save_pretrained(
                os.path.join(
                    self.model_save_dir,
                    f"model-{self.current_epoch:02d}epoch-{self.global_step}steps-{val_loss_mean:.4f}loss-{val_acc_mean:.4f}acc",
                ),
            )
