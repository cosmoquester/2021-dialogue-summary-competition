import os
import tempfile

import pytest
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from summarizer.method import ReinforceLearningModule

from ..constant import TOKENIZER_PATH
from .test_default import BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN, SUMMARY_MAX_SEQ_LEN, VOCAB_SIZE, config


@pytest.fixture()
def module():
    model = BartForConditionalGeneration(config)
    total_steps = 100
    max_learning_rate = 2e-4
    min_learning_rate = 1e-5
    warmup_rate = 0.06
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    summary_max_length = 20
    alpha = 0.9984

    import MeCab

    tagger = MeCab.Tagger
    MeCab.Tagger = type("Tagger", (object,), {"parse": lambda _, x: "\n".join(t + "\tDUMMY" for t in x.split())})

    with tempfile.TemporaryDirectory() as model_save_dir:
        yield ReinforceLearningModule(
            model,
            total_steps,
            max_learning_rate,
            min_learning_rate,
            warmup_rate,
            model_save_dir,
            tokenizer,
            summary_max_length,
            alpha,
        )

    MeCab.Tagger = tagger


def test_training_step(module: ReinforceLearningModule):
    batch = {
        "input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN)),
        "attention_mask": torch.ones((BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN), dtype=torch.float32),
        "decoder_input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
        "decoder_attention_mask": torch.ones((BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
    }

    module.training_step(batch, 0)


def test_validation_step(module: ReinforceLearningModule):
    batch = {
        "input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN)),
        "attention_mask": torch.ones((BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN), dtype=torch.float32),
        "decoder_input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
        "decoder_attention_mask": torch.ones((BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
    }

    module.validation_step(batch, 0)


def test_configure_optimizers(module: ReinforceLearningModule):
    module.configure_optimizers()


def test_validation_epoch_end(module: ReinforceLearningModule):
    module.all_gather = lambda x: x
    module.trainer = type("Trainer", (object,), {"is_global_zero": True, "current_epoch": 1, "global_step": 100})
    module.validation_epoch_end([{"val_loss": torch.tensor(0.1234), "val_rouge_l": torch.tensor(0.9999)}] * 10)

    assert os.listdir(module.model_save_dir)
