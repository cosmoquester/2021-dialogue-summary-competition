import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import BartForConditionalGeneration

from ..scheduler import LinearWarmupLR


class RDropModule(pl.LightningModule):
    """pytorch lightning module to train BART for dialogue summarization using RDrop
    https://arxiv.org/abs/2106.14448

    Attributes:
        model: BART model
        total_steps: total training steps for lr scheduling
        max_learning_rate: Max LR
        min_learning_rate: Min LR
        warmup_rate: warmup step rate
        model_save_dir: path to save model
        rdrop_alpha: R-Drop parameter
    """

    def __init__(
        self,
        model: BartForConditionalGeneration,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: float,
        model_save_dir: str,
        rdrop_alpha: float = 0.7,
    ):
        super().__init__()

        self.save_hyperparameters(
            {
                **model.config.to_dict(),
                "total_steps": total_steps,
                "max_learning_rate": max_learning_rate,
                "min_learning_rate": min_learning_rate,
                "rdrop_alpha": rdrop_alpha,
            }
        )

        self.model = model
        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_rate = warmup_rate
        self.model_save_dir = model_save_dir
        self.rdrop_alpha = rdrop_alpha

    def training_step(self, batch, batch_idx):
        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        output1 = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )
        logits1 = output1["logits"][:, :-1].reshape([labels.shape[0], -1])
        probs1 = F.softmax(logits1, dim=-1).gather(dim=1, index=labels.unsqueeze(1))
        log_probs1 = probs1.log()
        ce_loss1 = -log_probs1.mean()
        accuracy = torchmetrics.functional.accuracy(logits1, labels, ignore_index=self.model.config.pad_token_id)

        if self.rdrop_alpha == 0.0:
            metrics = {"loss": ce_loss1, "acc": accuracy}
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)
            return metrics

        output2 = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )
        logits2 = output2["logits"][:, :-1].reshape([labels.shape[0], -1])
        probs2 = F.softmax(logits2, dim=-1).gather(dim=1, index=labels.unsqueeze(1))
        log_probs2 = probs2.log()
        ce_loss2 = -log_probs2.mean()

        ce_loss = ce_loss1 + ce_loss2

        kl_div1 = F.kl_div(probs1, probs2)
        kl_div2 = F.kl_div(probs2, probs1)
        kl_div = (kl_div1 + kl_div2) / 2

        loss = ce_loss + self.rdrop_alpha * kl_div

        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_div": kl_div, "acc": accuracy}
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
