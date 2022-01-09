import argparse
import os
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

from summarizer.data import DialogueSummarizationDataset, PretrainDataset
from summarizer.method import DefaultModule, R3FModule, RDropModule, ReinforceLearningModule
from summarizer.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Train Dialogue Summarization with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--method", type=str, choices=["default", "rdrop", "r3f", "rl", "pretrain"], default="default", help="training method")
g.add_argument("--model-config-path", type=str, help="model config file path")
g.add_argument("--tokenizer", type=str, required=True, help="huggingface pretrained tokenizer path")
g.add_argument("--train-dataset-pattern", type=str, required=True, help="glob pattern of train dataset files")
g.add_argument("--valid-dataset-pattern", type=str, required=True, help="glob pattern of valid dataset files")
g.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
g.add_argument("--batch-size", type=int, default=128, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=256, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.05, help="warmup step rate")
g.add_argument("--dialogue-max-seq-len", type=int, default=256, help="dialogue max sequence length")
g.add_argument("--summary-max-seq-len", type=int, default=64, help="summary max sequence length")
g.add_argument("--gpus", type=int, default=0, help="the number of gpus")
g.add_argument("--all-dropout", type=float, help="override all dropout")
g.add_argument("--logging-interval", type=int, default=100, help="logging interval")
g.add_argument("--evaluate-interval", type=int, default=500, help="validation interval")
g.add_argument("--seed", type=int, default=42, help="random seed")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, help="wanDB entity name")
g.add_argument("--wandb-project", type=str, help="wanDB project name")

g = parser.add_argument_group("Method Specific Parameter")
g.add_argument("--rdrop-alpha", type=float, default=0.7, help="rdrop alpha parameter (only used with `rdrop` method)")
g.add_argument("--r3f-lambda", type=float, default=1.0, help="r3f lambda parameter (only used with `r3f` method)")
g.add_argument("--rl-alpha", type=float, default=0.9984, help="rl alpha parameter (only used with `rl` method)")
g.add_argument("--masking-rate", type=float, default=0.3, help="pretrain parameter (only used with `pretrain` method)")
# fmt: on


def main(args: argparse.Namespace):
    logger = get_logger("train")

    os.makedirs(args.output_dir)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.method == "pretrain":
        logger.info(f'[+] Load Train Dataset from "{args.train_dataset_pattern}"')
        train_dataset = PretrainDataset(
            paths=glob(args.train_dataset_pattern),
            tokenizer=tokenizer,
            dialogue_max_seq_len=args.dialogue_max_seq_len,
            masking_rate=args.masking_rate,
        )
        logger.info(f'[+] Load Valid Dataset from "{args.valid_dataset_pattern}"')
        valid_dataset = PretrainDataset(
            paths=glob(args.valid_dataset_pattern),
            tokenizer=tokenizer,
            dialogue_max_seq_len=args.dialogue_max_seq_len,
            masking_rate=args.masking_rate,
        )
    else:
        logger.info(f'[+] Load Train Dataset from "{args.train_dataset_pattern}"')
        train_dataset = DialogueSummarizationDataset(
            paths=glob(args.train_dataset_pattern),
            tokenizer=tokenizer,
            dialogue_max_seq_len=args.dialogue_max_seq_len,
            summary_max_seq_len=args.summary_max_seq_len,
            use_summary=True,
        )
        logger.info(f'[+] Load Valid Dataset from "{args.valid_dataset_pattern}"')
        valid_dataset = DialogueSummarizationDataset(
            paths=glob(args.valid_dataset_pattern),
            tokenizer=tokenizer,
            dialogue_max_seq_len=args.dialogue_max_seq_len,
            summary_max_seq_len=args.summary_max_seq_len,
            use_summary=True,
        )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    total_steps = len(train_dataloader) * args.epochs

    override_args = (
        {
            "dropout": args.all_dropout,
            "attention_dropout": args.all_dropout,
            "activation_dropout": args.all_dropout,
            "classifier_dropout": args.all_dropout,
        }
        if args.all_dropout
        else {}
    )
    if args.pretrained_ckpt_path:
        logger.info(f'[+] Load Model from "{args.pretrained_ckpt_path}"')
        model = BartForConditionalGeneration.from_pretrained(args.pretrained_ckpt_path, **override_args)
    elif args.model_config_path:
        logger.info(f'[+] Initialize Model using config from "{args.model_config_path}"')
        model = BartForConditionalGeneration(BartConfig.from_pretrained(args.model_config_path, **override_args))
    else:
        raise ValueError("[-] `--model-config-path` or `--pretrained-ckpt-path` is required!")

    logger.info(f"[+] Use method: {args.method}")
    model_dir = os.path.join(args.output_dir, "models")
    if args.method == "rdrop":
        lightning_module = RDropModule(
            model,
            total_steps,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_rate,
            model_dir,
            args.rdrop_alpha,
        )
    elif args.method == "r3f":
        lightning_module = R3FModule(
            model,
            total_steps,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_rate,
            model_dir,
            args.r3f_lambda,
        )
    elif args.method == "rl":
        lightning_module = ReinforceLearningModule(
            model,
            total_steps,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_rate,
            model_dir,
            tokenizer,
            args.summary_max_seq_len,
            args.rl_alpha,
        )
    else:
        lightning_module = DefaultModule(
            model, total_steps, args.max_learning_rate, args.min_learning_rate, args.warmup_rate, model_dir
        )

    logger.info(f"[+] Start Training")
    train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
    if args.wandb_project:
        train_loggers.append(
            WandbLogger(
                name=args.wandb_run_name or os.path.basename(args.output_dir),
                project=args.wandb_project,
                entity=args.wandb_entity,
                save_dir=args.output_dir,
            )
        )
    trainer = pl.Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        gpus=args.gpus,
    )
    trainer.fit(lightning_module, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
