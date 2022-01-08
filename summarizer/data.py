import csv
import json
import random
from typing import Dict, List, Optional, Tuple

import torch
from transformers.tokenization_utils import PreTrainedTokenizerBase


def load_json_data(path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Load dialogue summarization dataset json files of https://aihub.or.kr/aidata/30714

    Args:
        path: path of json file
    Returns:
        result of file, which is a tuple of ids, dialogues, summaries
    """
    with open(path) as f:
        data = json.load(f)

    ids = []
    dialogues = []
    summaries = []
    for datum in data["data"]:
        ids.append(datum["header"]["dialogueInfo"]["dialogueID"])

        prev_speaker_id = None
        prev_line = ""
        utts = []
        for dialogue in datum["body"]["dialogue"]:
            utterance = dialogue["utterance"].strip()

            if dialogue["participantID"] == prev_speaker_id:
                prev_line += " " + utterance
            else:
                if prev_line:
                    utts.append(prev_line)
                prev_line = utterance
                prev_speaker_id = dialogue["participantID"]
        if prev_line:
            utts.append(prev_line)

        dialogues.append(utts)
        summaries.append(datum["body"].get("summary"))
    return ids, dialogues, summaries


def load_tsv_data(path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Load arbitrary tsv file of formed like (id, dialogue, summary) with header
    each `dialogue` should be dumped json string from a list of utterances.
    ex) '["안녕", "잘가"]'

    Args:
        path: path of tsv file
    Returns:
        result of file, which is a tuple of ids, dialogues, summaries
    """
    ids = []
    dialogues = []
    summaries = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids.append(row["id"])
            dialogues.append(json.loads(row["dialogue"]))
            summaries.append(row.get("summary"))
    return ids, dialogues, summaries


class DialogueSummarizationDataset(torch.utils.data.Dataset):
    """Dataset for Dialogue Summarization

    Attributes:
        sep_token: token to seperate utterances
        ids: id of each example
        dialogues: dialogue of each example
        summaries: summary of each example
        dialogue_input_ids: dialogue input id tokens of each example
        dialogue_attention_masks: dialogue attention masks of each example
        summary_input_ids: summary input id tokens of each example
        summary_attention_masks: summary attention masks of each example
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        use_summary: bool,
    ):
        """
        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            summary_max_seq_len: max sequence length of summary
            use_summary: whether to use summary data or not (should be False for inference)
        """
        super().__init__()

        self.sep_token = tokenizer.sep_token
        (
            self.ids,
            self.dialogues,
            self.summaries,
            self.dialogue_input_ids,
            self.dialogue_attention_masks,
            self.summary_input_ids,
            self.summary_attention_masks,
        ) = self.load_dataset(paths, tokenizer, dialogue_max_seq_len, summary_max_seq_len, use_summary)

    def load_dataset(
        self,
        paths: List[str],
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        use_summary: bool,
    ) -> Tuple[
        List[str],
        List[List[str]],
        List[str],
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        """Load dataset files and featurize with tokenizer

        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            summary_max_seq_len: max sequence length of summary
            use_summary: whether to use summary data or not (should be False for inference)
        Returns:
            original ids, dialogues, summaries and input ids and attention masks for dialogues and summaries
        """
        ids, dialogues, summaries = [], [], []
        for path in paths:
            loader_fn = load_tsv_data if path.endswith(".tsv") else load_json_data

            file_ids, file_dialogues, file_summaries = loader_fn(path)
            ids.extend(file_ids)
            dialogues.extend(self.sep_token.join(x) for x in file_dialogues)
            summaries.extend(file_summaries)

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        dialogue_inputs = tokenizer(
            [bos + x + eos for x in dialogues],
            padding="max_length",
            truncation=True,
            max_length=dialogue_max_seq_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        summary_inputs = (
            tokenizer(
                [bos + x + eos for x in summaries],
                padding="max_length",
                truncation=True,
                max_length=summary_max_seq_len,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            if use_summary
            else {}
        )

        return (
            ids,
            dialogues,
            summaries,
            dialogue_inputs["input_ids"],
            dialogue_inputs["attention_mask"],
            summary_inputs.get("input_ids"),
            summary_inputs.get("attention_mask"),
        )

    def __len__(self) -> int:
        return len(self.dialogue_input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {"input_ids": self.dialogue_input_ids[index], "attention_mask": self.dialogue_attention_masks[index]}
        if self.summary_input_ids is not None and self.summary_attention_masks is not None:
            item.update(
                {
                    "decoder_input_ids": self.summary_input_ids[index],
                    "decoder_attention_mask": self.summary_attention_masks[index],
                }
            )
        return item


class PretrainDataset(torch.utils.data.Dataset):
    """Dataset for pretraining of BART with dialogue

    Attributes:
        tokenizer: tokenizer to tokenize dialogue and summary string
        dialogue_max_seq_len: max sequence length of dialouge
        masking_rate: rate of the number of masked token / sequence length
        bos_token: bos token
        eos_token: eos token
        sep_token: turn seperation token to divide each utterances
        mask_token_id: mask token id for text infilling
        ids: id of each example
        dialogues: dialogue of each example
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        masking_rate: float = 0.3,
    ):
        """
        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            masking_rate: rate of the number of masked token / sequence length
        Returns:
            original ids, dialogues, summaries and input ids and attention masks for dialogues and summaries
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.dialogue_max_seq_len = dialogue_max_seq_len
        self.masking_rate = masking_rate
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.sep_token = tokenizer.sep_token
        self.mask_token_id = tokenizer.mask_token_id
        self.ids, self.dialogues = self.load_dataset(paths)

    def load_dataset(self, paths: List[str]) -> Tuple[List[str], List[List[str]]]:
        """Load dataset files and featurize with tokenizer

        Args:
            paths: list of dataset paths (tsv or json)
        Returns:
            original ids, and sep token joined dialogues
        """
        ids, dialogues = [], []
        for path in paths:
            loader_fn = load_tsv_data if path.endswith(".tsv") else load_json_data

            file_ids, file_dialogues, _ = loader_fn(path)
            ids.extend(file_ids)
            dialogues.extend(file_dialogues)

        return ids, dialogues

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dialogue = self.dialogues[index]

        # Permutate
        random.shuffle(dialogue)

        # Tokenize
        dialogue_input = self.tokenizer(
            self.bos_token + self.sep_token.join(dialogue) + self.eos_token,
            padding="max_length",
            truncation=True,
            max_length=self.dialogue_max_seq_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        decoder_input_ids = dialogue_input["input_ids"][0]
        decoder_attention_mask = dialogue_input["attention_mask"][0]
        encoder_input_ids = decoder_input_ids.clone()
        encoder_attention_mask = decoder_attention_mask.clone()

        # Masking
        sequence_length = encoder_attention_mask.sum()
        num_masking = int(sequence_length * self.masking_rate)
        indices = torch.randperm(sequence_length)[:num_masking]
        encoder_input_ids[indices] = self.mask_token_id

        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
