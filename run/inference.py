import argparse
import csv
from glob import glob

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from summarizer.data import DialogueSummarizationDataset
from summarizer.utils import get_logger

parser = argparse.ArgumentParser(prog="inference", description="Inference Dialogue Summarization with BART")
parser.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
parser.add_argument("--tokenizer", type=str, required=True, help="huggingface pretrained tokenizer path")
parser.add_argument("--dataset-pattern", type=str, required=True, help="glob pattern of inference dataset files")
parser.add_argument("--output-path", type=str, required=True, help="output tsv file path")
parser.add_argument("--batch-size", type=int, default=512, help="inference batch size")
parser.add_argument("--dialogue-max-seq-len", type=int, default=256, help="dialogue max sequence length")
parser.add_argument("--summary-max-seq-len", type=int, default=64, help="summary max sequence length")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--length-penalty", type=float, default=1.2, help="beam search length penalty")
parser.add_argument("--device", type=str, default="cpu", help="inference device")


def main(args: argparse.Namespace):
    logger = get_logger("inference")

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)

    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Dataset from "{args.dataset_pattern}"')
    dataset_files = glob(args.dataset_pattern)

    logger.info(f'[+] Found Datasets: {", ".join(dataset_files)}')
    dataset = DialogueSummarizationDataset(
        paths=dataset_files,
        tokenizer=tokenizer,
        dialogue_max_seq_len=args.dialogue_max_seq_len,
        summary_max_seq_len=args.summary_max_seq_len,
        use_summary=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    logger.info(f'[+] Load Model from "{args.pretrained_ckpt_path}"')
    model = BartForConditionalGeneration.from_pretrained(args.pretrained_ckpt_path)
    model.to(device)

    logger.info("[+] Eval mode & Disable gradient")
    model.eval()
    torch.set_grad_enabled(False)

    logger.info("[+] Start Inference")
    total_summary_tokens = []
    for batch in tqdm(dataloader):
        dialoge_input = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # [BatchSize, SummarySeqLen]
        summary_tokens = model.generate(
            dialoge_input,
            attention_mask=attention_mask,
            decoder_start_token_id=tokenizer.bos_token_id,
            max_length=args.summary_max_seq_len,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            use_cache=True,
        )
        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())

    logger.info("[+] Start Decoding")
    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]

    logger.info(f'[+] Save Output to "{args.output_path}"')
    with open(args.output_path, "w") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["id", "dialogue", "target summary", "predict summary"])

        for row in zip(dataset.ids, dataset.dialogues, dataset.summaries, decoded):
            writer.writerow(row)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
