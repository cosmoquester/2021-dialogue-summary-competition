import csv
import json
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer


def load_json_data(path: str, sept_token: str) -> Tuple[List[str], List[str], List[str]]:
    """Load dialogue summarization dataset json files of https://aihub.or.kr/aidata/30714

    Args:
        path: path of json file
        sept_token: turn seperation token to divide each utterances
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

        dialogues.append(sept_token.join(utts))
        summaries.append(datum["body"].get("summary"))
    return ids, dialogues, summaries


def load_tsv_data(path: str, sept_token: str) -> Tuple[List[str], List[str], List[str]]:
    """Load arbitrary tsv file of formed like (id, dialogue, summary) with header
    each `dialogue` should be dumped json string from a list of utterances.
    ex) '["안녕", "잘가"]'

    Args:
        path: path of tsv file
        sept_token: turn seperation token to divide each utterances
    Returns:
        result of file, which is a tuple of ids, dialogues, summaries
    """
    ids = []
    dialogues = []
    summaries = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            dialogue = json.loads(row["dialogue"])

            ids.append(row["id"])
            dialogues.append(sept_token.join(dialogue))
            summaries.append(row.get("summary"))
    return ids, dialogues, summaries


class DialogueSummarizationDataset(torch.utils.data.Dataset):
    """Dataset for Dialogue Summarization

    Attributes:
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
        tokenizer: AutoTokenizer,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        sept_token: str,
        use_summary: bool,
    ):
        """
        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            summary_max_seq_len: max sequence length of summary
            sept_token: turn seperation token to divide each utterances
            use_summary: whether to use summary data or not (should be False for inference)
        """
        super().__init__()

        (
            self.ids,
            self.dialogues,
            self.summaries,
            self.dialogue_input_ids,
            self.dialogue_attention_masks,
            self.summary_input_ids,
            self.summary_attention_masks,
        ) = self.load_dataset(paths, tokenizer, dialogue_max_seq_len, summary_max_seq_len, sept_token, use_summary)

    def load_dataset(
        self,
        paths: List[str],
        tokenizer: AutoTokenizer,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        sept_token: str,
        use_summary: bool,
    ) -> Tuple[
        List[str],
        List[str],
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
            sept_token: turn seperation token to divide each utterances
            use_summary: whether to use summary data or not (should be False for inference)
        Returns:
            original ids, dialogues, summaries and input ids and attention masks for dialogues and summaries
        """
        ids, dialogues, summaries = [], [], []
        for path in paths:
            loader_fn = load_tsv_data if path.endswith(".tsv") else load_json_data

            file_ids, file_dialogues, file_summaries = loader_fn(path, sept_token)
            ids.extend(file_ids)
            dialogues.extend(file_dialogues)
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
