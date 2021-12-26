import argparse
import os
import tempfile
from glob import glob

import httpimport
import sentencepiece as spm
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(prog="train_tokenizer", description="Training Huggingface Tokenizer")
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
    "#@시스템#사진#",
    "#@시스템#동영상#",
    "#@시스템#기타#",
    "#@시스템#검색#",
    "#@시스템#지도#",
    "#@시스템#삭제#",
    "#@시스템#파일#",
    "#@시스템#송금#",
    "#@시스템#",
]

SENTENCEPIECE_URI = "https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece"

PAD = "[PAD]"
UNK = "[UNK]"
BOS = "[BOS]"
EOS = "[EOS]"
MASK = "[MASK]"
SEP = "[SEP]"


def main(args: argparse.Namespace):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "tokenizer")

        spm.SentencePieceTrainer.train(
            input=",".join(glob(args.data_pattern)),
            model_prefix=model_prefix,
            model_type="unigram",
            vocab_size=args.vocab_size,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=PAD,
            unk_piece=UNK,
            bos_piece=BOS,
            eos_piece=EOS,
            user_defined_symbols=[MASK, SEP, *special_words],
        )

        with httpimport.remote_repo(["sentencepiece_model_pb2"], SENTENCEPIECE_URI):
            import sentencepiece_model_pb2

            tokenizer = SentencePieceUnigramTokenizer.from_spm(model_prefix + ".model")

    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS,
        eos_token=EOS,
        cls_token=BOS,
        unk_token=UNK,
        sep_token=SEP,
        pad_token=PAD,
        mask_token=MASK,
        additional_special_tokens=special_words,
    )
    pretrained_tokenizer.save_pretrained(args.tokenizer_path)
    print(f"[+] Saved to {args.tokenizer_path}")


if __name__ == "__main__":
    exit(main(parser.parse_args()))
