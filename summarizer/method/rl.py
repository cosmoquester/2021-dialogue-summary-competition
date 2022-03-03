import os
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import AutoTokenizer, BartForConditionalGeneration, MaxLengthCriteria, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput

from ..scheduler import LinearWarmupLR


class ReinforceLearningModule(pl.LightningModule):
    """pytorch lightning module to train BART for dialogue summarization using RL

    Attributes:
        model: BART model
        total_steps: total training steps for lr scheduling
        max_learning_rate: Max LR
        min_learning_rate: Min LR
        warmup_rate: warmup step rate
        model_save_dir: path to save model
        tokenizer: tokenizer to decode output
        summary_max_length: summary max sequence length
        alpha: RL parameter
    """

    def __init__(
        self,
        model: BartForConditionalGeneration,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: float,
        model_save_dir: str,
        tokenizer: AutoTokenizer,
        summary_max_length: int,
        alpha: float = 0.9984,
    ):
        super().__init__()

        self.save_hyperparameters(
            {
                **model.config.to_dict(),
                "total_steps": total_steps,
                "max_learning_rate": max_learning_rate,
                "min_learning_rate": min_learning_rate,
                "summary_max_length": summary_max_length,
                "alpha": alpha,
            }
        )

        self.model: BartForConditionalGeneration = model
        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_rate = warmup_rate
        self.model_save_dir = model_save_dir
        self.tokenizer = tokenizer
        self.alpha = alpha

        self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(summary_max_length)])

        try:
            from MeCab import Tagger
        except ModuleNotFoundError:
            print("[-] Please install mecab and `mecab-python3` package to use RL Module")
            exit(1)

        try:
            from rouge_score.rouge_scorer import RougeScorer
        except ModuleNotFoundError:
            print("[-] Please `pip install rouge-score`")
            exit(1)

        mecab_dir = os.getenv("MECAB_DICT_DIR")
        if mecab_dir:
            self.tagger = Tagger(f"-d {mecab_dir}")
            print(f'[+] Use mecab dictionary "{mecab_dir}"')
        else:
            self.tagger = Tagger()
            print(f"[+] Use default mecab dictionary, Please set 'MECAB_DICT_DIR' environemt variable for custom dict")
        self.rouge = RougeScorer(["rougeL"])

    def to_morphemes(self, text: str) -> str:
        """Tokenize text with mecab and rejoin with white space"""
        tagged = self.tagger.parse(text).split("\n")
        tagged = [t.split("\t")[0] for t in tagged if "\t" in t]
        return " ".join(tagged)

    def get_rouge_f1(self, predictions: List[str], targets: List[str]) -> List[float]:
        """Calculate rouge-l f1 score"""
        targets = [self.to_morphemes(text) for text in targets]
        predictions = [self.to_morphemes(text) for text in predictions]
        return [
            self.rouge.score(target, prediction)["rougeL"].fmeasure for target, prediction in zip(targets, predictions)
        ]

    def training_step(self, batch, batch_idx):
        # Maximum likelihood
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )
        device = output.logits.device

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])

        ce_loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)
        accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=self.model.config.pad_token_id)

        # Policy Learning
        decoder_input_for_search = torch.full([output.logits.size(0), 1], self.model.config.bos_token_id, device=device)
        encoder_outputs = BaseModelOutput(last_hidden_state=output.encoder_last_hidden_state)
        with torch.no_grad():
            greedy_output = self.model.greedy_search(
                input_ids=decoder_input_for_search,
                encoder_outputs=encoder_outputs,
                return_dict_in_generate=True,
                stopping_criteria=self.stopping_criteria,
            )
        sampled_output = self.model.sample(
            input_ids=decoder_input_for_search,
            encoder_outputs=encoder_outputs,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=self.stopping_criteria,
        )

        # [BatchSize, SequenceLength, VocabSize]
        scores = torch.stack(sampled_output.scores, dim=1)
        log_probs = scores.log_softmax(dim=2)

        # [BatchSize, SequenceLength]
        log_probs = (log_probs * F.one_hot(sampled_output.sequences[:, 1:], log_probs.shape[2])).sum(dim=2)
        # [BatchSize]
        sequence_log_probs = log_probs.mean(dim=1)

        # Rouge difference of greedy and sample
        greedy_summaries = self.tokenizer.batch_decode(greedy_output.sequences)
        sampled_summaries = self.tokenizer.batch_decode(sampled_output.sequences)
        target_summaries = self.tokenizer.batch_decode(batch["decoder_input_ids"])

        greedy_rouge_l_f1 = self.get_rouge_f1(greedy_summaries, target_summaries)
        sampled_rouge_l_f1 = self.get_rouge_f1(sampled_summaries, target_summaries)

        greedy_rouge_l_f1 = torch.tensor(greedy_rouge_l_f1, dtype=torch.float32, device=device).detach()
        sampled_rouge_l_f1 = torch.tensor(sampled_rouge_l_f1, dtype=torch.float32, device=device).detach()

        rouge_diff = greedy_rouge_l_f1 - sampled_rouge_l_f1
        scaled_log_probs = sequence_log_probs * rouge_diff

        rl_loss = scaled_log_probs.mean(dim=0)

        loss = self.alpha * rl_loss + (1 - self.alpha) * ce_loss
        rouge_l_f1 = sum(greedy_rouge_l_f1) / len(greedy_rouge_l_f1)
        metrics = {"loss": loss, "ce_loss": ce_loss, "rl_loss": rl_loss, "acc": accuracy, "rouge-l-f1": rouge_l_f1}
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

        decoder_input_for_search = torch.full(
            [output.logits.size(0), 1], self.model.config.bos_token_id, device=logits.device
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=output.encoder_last_hidden_state)

        greedy_output = self.model.greedy_search(
            input_ids=decoder_input_for_search,
            encoder_outputs=encoder_outputs,
            return_dict_in_generate=True,
            stopping_criteria=self.stopping_criteria,
        )

        greedy_summaries = self.tokenizer.batch_decode(greedy_output.sequences)
        target_summaries = self.tokenizer.batch_decode(batch["decoder_input_ids"])

        greedy_rouge_l_f1 = self.get_rouge_f1(greedy_summaries, target_summaries)
        rouge_l_f1 = sum(greedy_rouge_l_f1) / len(greedy_rouge_l_f1)

        metrics = {"val_loss": loss, "val_acc": accuracy, "val_rouge_l": rouge_l_f1}
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
            val_rouge_ls = [output["val_rouge_l"].mean() for output in outputs]

            val_loss_mean = sum(val_losses) / len(val_losses)
            val_rouge_l_mean = sum(val_rouge_ls) / len(val_rouge_ls)

            self.model.save_pretrained(
                os.path.join(
                    self.model_save_dir,
                    f"model-{self.current_epoch:02d}epoch-{self.global_step}steps-{val_loss_mean:.4f}loss-{val_rouge_l_mean:.4f}rouge-l",
                ),
            )
