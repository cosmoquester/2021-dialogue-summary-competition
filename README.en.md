# 2021 Dialogue Summary Competition

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/2021-dialogue-summary-competition.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/2021-dialogue-summary-competition)
[![codecov](https://codecov.io/gh/cosmoquester/2021-dialogue-summary-competition/branch/master/graph/badge.svg?token=Vaq4tqh4TL)](https://codecov.io/gh/cosmoquester/2021-dialogue-summary-competition)

[[2021 Korean Voice/Natural Language AI Competition](http://aihub-competition.or.kr/hangeul)] This is repository to share training and inference codes of Team 알라꿍달라꿍.

Team: Sangjun Park([cosmoquester](https://github.com/cosmoquester)), Kiwon Choi([ckw1140](https://github.com/ckw1140)), Hyerin Oh([Hyerin-oh](https://github.com/Hyerin-oh))

Our team consists of these three people.

The final leaderboard is as follows, [Won 1st place in the dialogue summary section and got the Naver Representative Award](https://www.msit.go.kr/bbs/view.do?sCode=user&mId=113&mPid=112&pageIndex=1&bbsSeqNo=94&nttSeqNo=3181143&searchOpt=ALL&searchTxt=%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C)


| Rank | Name | Score |
| ---- | ---- | ----- |
| 1 | 알라꿍달라꿍 | 0.34286246 |
| 2 | ... | 0.3330445 |
| 3 | ... | 0.33254071 |

## Quick Start

As the technique used in the competition, you can easily use the model trained using only AIHub data without external data.

```sh
$ pip install transformers
```

```python
from transformers import pipeline

model_name = "alaggung/bart-r3f"
max_length = 64
dialogue = ["밥 ㄱ?", "고고고고 뭐 먹을까?", "어제 김치찌개 먹어서 한식말고 딴 거", "그럼 돈까스 어때?", "오 좋다 1시 학관 앞으로 오셈", "ㅇㅋ"]

summarizer = pipeline("summarization", model=model_name)
summarization = summarizer("[BOS]" + "[SEP]".join(dialogue) + "[EOS]", max_length=max_length)

print(summarization)
# Your max_length is set to 64, but you input_length is only 51. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=25)
# [{'summary_text': '어제 김치찌개를 먹어서 한식 말고 돈가스를 먹기로 했다.'}]
```
- You can run it with `pipeline` simply.

```python
from transformers import AutoTokenizer, BartForConditionalGeneration

model_name = "alaggung/bart-r3f"
max_length = 64
num_beams = 5
length_penalty = 1.2
dialogue = ["밥 ㄱ?", "고고고고 뭐 먹을까?", "어제 김치찌개 먹어서 한식말고 딴 거", "그럼 돈까스 어때?", "오 좋다 1시 학관 앞으로 오셈", "ㅇㅋ"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

inputs = tokenizer("[BOS]" + "[SEP]".join(dialogue) + "[EOS]", return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=num_beams,
    length_penalty=length_penalty,
    max_length=max_length,
    use_cache=True,
)
summarization = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summarization)
# 어제 김치찌개를 먹어서 한식 말고 돈가스를 먹기로 했다.
```
- You can make an inference manually like this.

## Directory Structure

```
# The run directory contains executable scripts.
run
├── inference.py
├── interactive.py
├── train.py
└── train_tokenizer.py

# The summarizer directory contains data loader and training logic.
summarizer
├── data.py
├── method
│   ├── default.py
│   ├── r3f.py
│   ├── rdrop.py
│   └── rl.py
├── scheduler.py
└── utils.py
```

## Process

The procedure for learning the dialogue summary model used in the final submission is as follows. In the attached running example, the hyperparameters are actually the same as those used in the competition. These parameters were set on one V100 32GB GPU environment. You may need to be adjusted in other environments.

### 1. Tokenizer training

Our team trained tokenizer directly from the training data of the competition.
To learn the tokenizer, the utterances and summaries of the data were first segmented with mecab, and then the unigram method was used. (See **run/train_tokenizer.py**)

### 2. BART pretraining

We used BART as our model architecture.
Since external data was not allowed, pre-training was performed with the dialogue text of the training dataset, and text infilling and sentence permutation were applied for noise.
Sentence permutation was performed on a turn-by-turn basis.

```sh
$ python -m run train \
    --output-dir outputs/pretrain \
    --method pretrain \
    --tokenizer resources/tokenizers/unigram_4K \
    --train-dataset-pattern "data/Training/*.json" \
    --valid-dataset-pattern "data/Validation/*.json" \
    --epochs 50 --seed 42 --max-learning-rate 2e-4 --batch-size 64 --gpus 1 \
    --model-config-path resources/configs/default.json
```
- The training detail of this learning can be viewed in [Wandb](https://wandb.ai/alaggung/dialogue_summarization_public/runs/2vn8htcd).
- The model trained in this step can be used in [alaggung/bart-pretrained](https://huggingface.co/alaggung/bart-pretrained).

### 3. Dialogue Summarization finetune (R3F)

After pre-training, a Dialogue Summarization task was trained.
In this case, the R3F technique, which shows a good performance in Abstract Summarization, was applied.

```sh
$ python -m run train \
    --output-dir outputs/r3f \
    --method r3f \
    --tokenizer resources/tokenizers/unigram_4K \
    --train-dataset-pattern "data/Training/*.json" \
    --valid-dataset-pattern "data/Validation/*.json" \
    --epochs 10 --seed 42 --batch-size 64 --max-learning-rate 2e-4 --gpus 1 \
    --model-config-path resources/configs/default.json \
    --pretrained-ckpt-path outputs/pretrain/models/model-49epoch-218374steps-0.6568loss-0.8601acc
```
- The training detail of this learning can be viewed in [Wandb](https://wandb.ai/alaggung/dialogue_summarization_public/runs/2yc359fn).
- The model trained in this step can be used in [alaggung/bart-r3f](https://huggingface.co/alaggung/bart-r3f).

### 4. Dialogue Summarization finetune (RL)

Finally, classical RL was applied to align the learning goal with the ROUGE-L F1 score, which is the evaluation metric of the competition.
As the target metric, the ROUGE-L F1 score was used based on the mecab segmentation between the model-generated summary and the actual summary.

```sh
$ python -m run train \
    --output-dir outputs/rl \
    --method rl \
    --tokenizer resources/tokenizers/unigram_4K \
    --train-dataset-pattern "data/Training/*.json" \
    --valid-dataset-pattern "data/Validation/*.json" \
    --epochs 1 --seed 42 --max-learning-rate 2e-5 --batch-size 20 --valid-batch-size 32 --accumulate-grad-batches 6 --gpus 1 \
    --model-config-path resources/configs/default.json \
    --pretrained-ckpt-path outputs/r3f/models/model-09epoch-43374steps-1.2955loss-0.6779acc
```
- The training detail of this learning can be viewed in [Wandb](https://wandb.ai/alaggung/dialogue_summarization_public/runs/3ae2abvk).
- The model trained in this step can be used in [alaggung/bart-rl](https://huggingface.co/alaggung/bart-rl).

## Run

### Train

```sh
$ python -m run train \
    --output-dir outputs/default-training \
    --method default \
    --model-config-path resources/configs/default.json \
    --tokenizer resources/tokenizers/unigram_4K \
    --train-dataset-pattern "data/Training/*.json" \
    --valid-dataset-pattern "data/Validation/*.json" \
    --gpus 1
```
- As above, you can run the training script by giving the data, config, and tokenizer path.
- The `method` argument sets the training technique, and one of `default`, `pretrain`, `r3f`, `rdrop`, and `rl` should be selected.
- `pretrain` is a BART pretrain, so it does not teach the conversation summary task.
- `rl` calculates and uses the ROUGE-L F1 score in learning. At this time, since morphemes are segmented using mecab, mecab must be installed and it's better that the **docker/dialogue-summary-specials.csv** file was added to the mecab user dictionary.
- The training script saves checkpoints for every validation. All output is stored in the `--output-dir` directory and the models are stored in the models directory in the format of huggingface pretrained models.

```sh
$ docker run --rm \
    --runtime nvidia \
    -v `pwd`:/project \
    cosmoquester/2021-dialogue-summary-competition:latest-gpu train \
    --output-dir /project/outputs/default \
    --method default \
    --model-config-path /project/resources/configs/default.json \
    --tokenizer /project/resources/tokenizers/unigram_4K \
    --train-dataset-pattern "/project/data/Training/*.json" \
    --valid-dataset-pattern "/project/data/Validation/*.json" \
    --gpus 1
```
- If the process of downloading the code and installing the Python package or installing mecab and adding the user dictionary is bothersome, you can perform learning with docker as above, regardless of the environment. Of course, Inference or Interactive Test below can also be run with docker in the same way.
- Available docker images are [here](https://hub.docker.com/r/cosmoquester/2021-dialogue-summary-competition/tags).
- In case of using GPU, you need to set parameters such as `--runtime nvidia` or `--gpus all` according to the version of nvidia-docker. You should also use tags with `-gpu`, such as `latest-gpu`.
- If you want to record the run in Wandb, you must either directly input the API key by giving the `-it` option or enter the API key as the `WANDB_API_KEY` environment variable.

### Inference

```sh
$ python -m run.inference \
    --pretrained-ckpt-path alaggung/bart-r3f \
    --tokenizer alaggung/bart-r3f \
    --dataset-pattern "data/Validation/*.json" \
    --output-path result.tsv \
    --device cuda
```
- Above code is to check the inference result of the model. Executing like this, the result inferred by the model is saved in result.tsv. tsv file consists of 4 columns: id, dialogue, target summary, predict summary.
- You can compare summary sentences qualitatively with the result file, or calculate and analyze scores such as ROUGE with target summary and predict summary.

### Interactive Test

```sh
$ python -m run interactive \
    --pretrained-ckpt-path alaggung/bart-r3f  \
    --tokenizer alaggung/bart-r3f \
    --device cuda
[2022-01-10 00:48:35,717] [+] Use Device: cuda
[2022-01-10 00:48:35,718] [+] Load Tokenizer from "alaggung/bart-r3f"
[2022-01-10 00:48:35,727] [+] Load Model from "alaggung/bart-r3f"
[2022-01-10 00:48:39,918] [+] Eval mode & Disable gradient
Start Interactive Summary? (Y/n) 
Utterance 1: 밥 ㄱ?
Utterance 2: 고고고고 뭐 먹을까?
Utterance 3: 어제 김치찌개 먹어서 한식말고 딴 거
Utterance 4: 그럼 돈까스 어때?
Utterance 5: 오 좋다 1시 학관 앞으로 오셈
Utterance 6: ㅇㅋ
Utterance 7: 
Summary:  어제 김치찌개를 먹어서 한식 말고 돈가스를 먹기로 했다.

Start Interactive Summary? (Y/n) n
```
- Interactive allows you to directly test the dialogue summary performance of the model while inputting utterances one by one. If you hit enter without entering an utterance, it will end and a summary will be printed.
- If you proceed with the summary, you will be asked if you want to start Interactive Summary again. If you want to quit, just enter no or n.

## References

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)
- [Better Fine-Tuning by Reducing Representational Collapse](https://arxiv.org/abs/2008.03156v1)
- [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)
