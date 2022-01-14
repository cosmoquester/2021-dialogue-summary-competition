# 2021 Dialogue Summary Competition

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/2021-dialogue-summary-competition.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/2021-dialogue-summary-competition)
[![codecov](https://codecov.io/gh/cosmoquester/2021-dialogue-summary-competition/branch/master/graph/badge.svg?token=Vaq4tqh4TL)](https://codecov.io/gh/cosmoquester/2021-dialogue-summary-competition)

Please [Click here](https://github.com/cosmoquester/2021-dialogue-summary-competition/blob/master/README.en.md) for English README. 

[[2021 훈민정음 한국어 음성•자연어 인공지능 경진대회](http://aihub-competition.or.kr/hangeul)] 대화요약 부문 알라꿍달라꿍 팀의 대화요약 학습 및 추론 코드를 공유하기 위한 레포입니다.

팀원: 박상준([cosmoquester](https://github.com/cosmoquester)), 최기원([ckw1140](https://github.com/ckw1140)), 오혜린([Hyerin-oh](https://github.com/Hyerin-oh))

저희 팀은 이렇게 세 명으로 구성되어 있습니다.

최종 리더보드는 아래와 같고, [대화요약 부문에서 1등을 해내 네이버 대표상을 수상](https://www.msit.go.kr/bbs/view.do?sCode=user&mId=113&mPid=112&pageIndex=1&bbsSeqNo=94&nttSeqNo=3181143&searchOpt=ALL&searchTxt=%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C)하였습니다.

| Rank | Name | Score |
| ---- | ---- | ----- |
| 1 | 알라꿍달라꿍 | 0.34286246 |
| 2 | ... | 0.3330445 |
| 3 | ... | 0.33254071 |

## Quick Start

대회에서 사용한 기법대로 외부데이터없이 AIHub 데이터만을 이용해 학습한 모델을 쉽게 사용해볼 수 있습니다.

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
- 위와 같이 `pipeline`을 이용하면 간단하게 실행할 수 있습니다.

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
- 이렇게 그냥 불러와 추론할 수도 있습니다.

## Directory Structure

```
# run 디렉토리에는 실행할 수 있는 스크립트들이 들어있습니다.
run
├── inference.py
├── interactive.py
├── train.py
└── train_tokenizer.py

# summarizer 디렉토리에는 데이터로더와 학습로직이 들어있습니다.
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

최종제출에 사용했던 대화요약 모델을 학습하는 절차는 다음과 같습니다. 첨부된 실행 예시에서 하이퍼파라미터는 실제로 대회에서 사용했던 것과 동일합니다. 해당 파라미터들은 V100 32GB GPU 1대 기준으로 타 환경에서는 조절이 필요할 수 있습니다.

### 1. Tokenizer training

저희 팀은 토크나이저를 대회에서 제공한 학습데이터로 직접 학습해 사용하였습니다.
토크나이저의 학습은 학습데이터의 발화와 요약문을 mecab으로 1차분절한 후에 unigram 방법으로 학습해 사용하였습니다. (**run/train_tokenizer.py** 참고)

### 2. BART pretraining

저희는 모델 아키텍쳐로는 BART를 사용했습니다.
외부 데이터를 사용할 수 없기 때문에 학습 데이터셋의 대화 텍스트로 사전학습을 수행했으며 노이즈는 Text infilling과 Sentence Permutation을 적용했습니다.
Sentence Permutation은 턴 단위로 수행하였습니다.

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
- 이 학습의 학습추이는 [Wandb](https://wandb.ai/alaggung/dialogue_summarization_public/runs/2vn8htcd)에서 볼 수 있습니다.
- 이 단계를 학습한 모델은 [alaggung/bart-pretrained](https://huggingface.co/alaggung/bart-pretrained)에서 사용할 수 있습니다.

### 3. Dialogue Summarization finetune (R3F)

사전학습 후에는 Dialogue Summarization task를 학습시켰습니다.
이때 Abstract Summarization에서 좋은 효과를 보이는 R3F 기법을 적용하였습니다.

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
- 이 학습의 학습추이는 [Wandb](https://wandb.ai/alaggung/dialogue_summarization_public/runs/2yc359fn)에서 볼 수 있습니다.
- 이 단계를 학습한 모델은 [alaggung/bart-r3f](https://huggingface.co/alaggung/bart-r3f)에서 사용할 수 있습니다.

### 4. Dialogue Summarization finetune (RL)

마지막으로 학습의 목표를 대회의 평가지표인 ROUGE-L F1 score와 align시키기 위해서 고전적인 RL을 적용했습니다.
target metric은 모델이 생성한 요약문과 실제 요약문간의 mecab분절 기준 ROUGE-L F1 score를 사용하였습니다.

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
- 이 학습의 학습추이는 [Wandb](https://wandb.ai/alaggung/dialogue_summarization_public/runs/3ae2abvk)에서 볼 수 있습니다.
- 이 단계를 학습한 모델은 [alaggung/bart-rl](https://huggingface.co/alaggung/bart-rl)에서 사용할 수 있습니다.

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
- 위와 같이 데이터와 config, tokenizer 경로를 주고 학습스크립트를 실행할 수 있습니다.
- `method` 인자는 학습 방법을 설정하는 것이며 `default`, `pretrain`, `r3f`, `rdrop`, `rl` 중에 하나를 택해야합니다. method에 따라 해당 학습 기법으로 학습합니다.
- `pretrain`은 BART pretrain이라서 대화요약 task를 학습하지 않습니다.
- `rl`은 학습에서 target summary와의 ROUGE-L F1 점수를 계산해 사용하는데 이때 mecab을 이용해 형태소를 분절하기 때문에 mecab이 설치되어야 하며 **docker/dialogue-summary-specials.csv** 파일을 사용자 사전에 추가해주면 좋습니다.
- 학습스크립트는 매 validation마다 체크포인트를 저장합니다. 모든 결과물은 `--output-dir` 디렉토리에 저장되며 모델은 huggingface pretrained 모델 형식으로 models 디렉토리에 저장됩니다.

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
- 코드를 다운받아서 파이썬 패키지를 설치하거나 mecab을 설치하고 사용자사전을 추가해주는 과정이 귀찮은 경우 위와 같이 docker로 학습을 실행하면 환경에 관계없이 편히 학습을 진행할 수 있습니다. 물론 아래에 있는 Inference나 Interactive Test도 같은 방식으로 docker로 실행할 수 있습니다.
- 사용가능한 도커이미지 목록은 [여기](https://hub.docker.com/r/cosmoquester/2021-dialogue-summary-competition/tags)에 있습니다.
- GPU를 사용하는 경우 nvidia-docker의 버전에 따라 `--runtime nvidia`나 `--gpus all` 등의 인자를 맞춰서 설정해주어야 합니다. 또한 `latest-gpu`와 같이 `-gpu`가 붙어있는 태그를 이용해야합니다.
- Wandb에 해당 run을 기록하고자하는 경우 `-it` 옵션을 줘서 직접 API키를 입력하거나 `WANDB_API_KEY` 환경변수로 API키를 넣어줘야 합니다.

### Inference

```sh
$ python -m run.inference \
    --pretrained-ckpt-path alaggung/bart-r3f \
    --tokenizer alaggung/bart-r3f \
    --dataset-pattern "data/Validation/*.json" \
    --output-path result.tsv \
    --device cuda
```
- 이 코드는 모델의 추론 결과를 확인하는 코드입니다. 이렇게 실행할 경우 해당 모델로 추론한 결과를 result.tsv에 저장합니다. tsv파일은 id, dialogue, target summary, predict summary 이렇게 4개의 열로 구성되어 있습니다.
- 결과 파일로 정성적으로 요약문을 비교해도 되고 target summary와 predict summary로 ROUGE 등의 점수를 계산해서 분석해도 됩니다.

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
- interactive는 발화를 하나씩 입력하면서 직접 모델의 대화요약 성능을 테스트해볼 수 있습니다. 발화를 입력하지 않고 엔터를 쳐서 넘기면 종료되고 요약문을 출력합니다.
- 요약을 진행하면 다시 Interactive Summary를 시작할 지 묻는데 종료하고 싶으면 no나 n을 입력하면 됩니다.

## References

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)
- [Better Fine-Tuning by Reducing Representational Collapse](https://arxiv.org/abs/2008.03156v1)
- [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)
