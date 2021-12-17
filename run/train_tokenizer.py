import argparse
from glob import glob

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import UnigramTrainer
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description="Training Huggingface Tokenizer")
parser.add_argument("--data-pattern", type=str, required=True, help="glob pattern for dataset files")
parser.add_argument("--tokenizer-path", type=str, required=True, help="path to save tokenizer")
parser.add_argument("--vocab-size", type=int, default=4000, help="vocab size of tokenizer")

special_words = [
    "#@주소#",
    "#@이모티콘#",
    "#@이름#",
    "#@URL#",
    "#@소속#",
    "#@기타#",
    "#@전번#",
    "#@계정#",
    "#@url#",
    "#@번호#",
    "#@금융#",
    "#@신원#",
    "#@장소#",
    "#@시스템#",
    "#@시스템#사진#",
    "#@시스템#동영상#",
    "#@시스템#기타#",
    "#@시스템#검색#",
    "#@시스템#지도#",
    "#@시스템#삭제#",
    "#@시스템#파일#",
    "#@시스템#송금#",
]


def main(args: argparse.Namespace):
    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    trainer = UnigramTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[SEP]", "[SEPT]", *special_words],
        unk_token="[UNK]",
    )

    print("[+] Start to train tokenizer")
    tokenizer.train(glob(args.data_pattern), trainer)

    print(f"[+] Save tokenizer to {args.tokenizer_path}")
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    pretrained_tokenizer.save_pretrained(args.tokenizer_path)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
