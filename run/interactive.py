import argparse

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from summarizer.utils import get_logger

parser = argparse.ArgumentParser(prog="interactive", description="Inference Dialogue Summarization with BART")
parser.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
parser.add_argument("--tokenizer", type=str, required=True, help="huggingface pretrained tokenizer path")
parser.add_argument("--dialogue-max-seq-len", type=int, default=256, help="dialogue max sequence length")
parser.add_argument("--summary-max-seq-len", type=int, default=64, help="summary max sequence length")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--length-penalty", type=float, default=1.2, help="beam search length penalty")
parser.add_argument("--device", type=str, default="cpu", help="inference device")


def main(args: argparse.Namespace):
    logger = get_logger("interactive")

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)

    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Model from "{args.pretrained_ckpt_path}"')
    model = BartForConditionalGeneration.from_pretrained(args.pretrained_ckpt_path)
    model.to(device)

    logger.info("[+] Eval mode & Disable gradient")
    model.eval()
    torch.set_grad_enabled(False)

    while True:
        if input("Start Interactive Summary? (Y/n) ").lower() in ("n", "no"):
            break

        utterances = []

        while True:
            inp = input(f"Utterance {len(utterances) + 1}: ")

            if inp == "":
                break

            utterances.append(inp)

        inputs = tokenizer(
            [tokenizer.bos_token + tokenizer.sep_token.join(utterances) + tokenizer.eos_token],
            return_tensors="pt",
            return_token_type_ids=False,
        )
        dialoge_input = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

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

        summary = tokenizer.decode(summary_tokens.squeeze().cpu().detach(), skip_special_tokens=True)
        print("Summary: ", summary)
        print()


if __name__ == "__main__":
    exit(main(parser.parse_args()))
