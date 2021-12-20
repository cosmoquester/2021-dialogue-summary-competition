import os
import tempfile

import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration

from summarizer.method import DefaultModule

VOCAB_SIZE = 28
DIALOGUE_MAX_SEQ_LEN = 20
SUMMARY_MAX_SEQ_LEN = 10
BATCH_SIZE = 2

config = BartConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=32,
    encoder_layers=1,
    encoder_ffn_dim=128,
    encoder_attention_heads=1,
    decoder_layers=1,
    decoder_ffn_dim=128,
    decoder_attention_heads=1,
    d_model=32,
)


@pytest.fixture()
def module():
    model = BartForConditionalGeneration(config)
    total_steps = 100
    max_learning_rate = 2e-4
    min_learning_rate = 1e-5
    warmup_rate = 0.06

    with tempfile.TemporaryDirectory() as model_save_dir:
        yield DefaultModule(model, total_steps, max_learning_rate, min_learning_rate, warmup_rate, model_save_dir)


def test_training_step(module: DefaultModule):
    batch = {
        "input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN)),
        "attention_mask": torch.ones((BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN), dtype=torch.float32),
        "decoder_input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
        "decoder_attention_mask": torch.ones((BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
    }

    module.training_step(batch, 0)


def test_validation_step(module: DefaultModule):
    batch = {
        "input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN)),
        "attention_mask": torch.ones((BATCH_SIZE, DIALOGUE_MAX_SEQ_LEN), dtype=torch.float32),
        "decoder_input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
        "decoder_attention_mask": torch.ones((BATCH_SIZE, SUMMARY_MAX_SEQ_LEN)),
    }

    module.validation_step(batch, 0)


def test_configure_optimizers(module: DefaultModule):
    module.configure_optimizers()


def test_validation_epoch_end(module: DefaultModule):
    module.all_gather = lambda x: x
    module.trainer = type("Trainer", (object,), {"is_global_zero": True, "current_epoch": 1, "global_step": 100})
    module.validation_epoch_end([{"val_loss": torch.tensor(0.1234), "val_acc": torch.tensor(0.9999)}] * 10)

    assert os.listdir(module.model_save_dir)
