import os

from transformers import AutoTokenizer

from summarizer.data import DialogueSummarizationDataset, load_json_data, load_tsv_data

from .constant import TEST_DATA_DIR, TOKENIZER_PATH

JSON_DATA_PATH = os.path.join(TEST_DATA_DIR, "sample.json")
TSV_DATA_PATH = os.path.join(TEST_DATA_DIR, "sample.tsv")
SEP = "[SEP]"


class DataInfo:
    ids = ["70b05b36-5f37-11ec-91f5-0c9d921fe15b", "8340450f-5f37-11ec-91f5-0c9d921fe15b"]
    dialogues = [
        "밥 먹었어~?[SEP]편의점에서 대충 때웠어[SEP]허어얼.. 편의점이라니[SEP]#@이모티콘#",
        "엄마가 잔뜩 사와가지고ㅋㅋㅋㅋ 주말에 먹을래??[SEP]싸오게??[SEP]남으면?? 오빠가 원하면[SEP]ㄴㄴ 괜찮은데[SEP]숨겨놓고 싸갈게 #@이모티콘#",
    ]
    summaries = ["편의점에서 대충 식사를 때웠다는 이야기를 하고 있다.", "엄마가 사 온 음식이 남아있으면 숨겨서라도 싸가겠다고 이야기한다."]


def test_load_json_data():
    ids, dialogues, summaries = load_json_data(JSON_DATA_PATH, SEP)

    assert len(ids) == len(dialogues) == len(summaries)
    assert ids == DataInfo.ids
    assert dialogues == DataInfo.dialogues
    assert summaries == DataInfo.summaries


def test_load_tsv_data():
    ids, dialogues, summaries = load_tsv_data(TSV_DATA_PATH, SEP)

    assert len(ids) == len(dialogues) == len(summaries)
    assert ids == DataInfo.ids
    assert dialogues == DataInfo.dialogues
    assert summaries == DataInfo.summaries


def test_dialogue_summarization_dataset():
    dialogue_max_seq_len = 20
    summary_max_seq_len = 10

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    dataset = DialogueSummarizationDataset(
        [JSON_DATA_PATH, TSV_DATA_PATH], tokenizer, dialogue_max_seq_len, summary_max_seq_len, True
    )

    assert len(dataset) == 4

    example = dataset[0]
    for key in ("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"):
        assert key in example

    assert len(example["input_ids"]) == len(example["attention_mask"]) == dialogue_max_seq_len
    assert len(example["decoder_input_ids"]) == len(example["decoder_attention_mask"]) == summary_max_seq_len
