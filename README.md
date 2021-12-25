# 2021 Dialogue Summary Competition

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/2021-dialogue-summary-competition.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/2021-dialogue-summary-competition)
[![codecov](https://codecov.io/gh/cosmoquester/2021-dialogue-summary-competition/branch/master/graph/badge.svg?token=Vaq4tqh4TL)](https://codecov.io/gh/cosmoquester/2021-dialogue-summary-competition)

[[2021 훈민정음 한국어 음성•자연어 인공지능 경진대회](http://aihub-competition.or.kr/hangeul)] 대화요약 부문 알라꿍달라꿍 팀의 대화요약 학습 및 추론 코드를 공유하기 위한 레포입니다.

팀원: 박상준([cosmoquester](https://github.com/cosmoquester)), 최기원([ckw1140](https://github.com/ckw1140)), 오혜린([Hyerin-oh](https://github.com/Hyerin-oh))

저희 팀은 이렇게 세 명으로 구성되어 있습니다.

최종 리더보드는 아래와 같고, [대화요약 부문에서 1등을 해내 네이버 대표상을 수상](https://www.msit.go.kr/bbs/view.do?sCode=user&mId=113&mPid=112&pageIndex=1&bbsSeqNo=94&nttSeqNo=3181143&searchOpt=ALL&searchTxt=%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C)하였습니다.

| Rank | Name | Score |
| ---- | ---- | ----- |
| 1 | 알라꿍달라꿍 | 0.34286246 |
| 2 | ... | 0.3330445 |
| 3 | ... | 0.33254071 |

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

최종제출에 사용했던 대화요약 모델을 학습하는 절차는 다음과 같습니다.

### 1. Tokenizer training

저희 팀은 토크나이저를 대회에서 제공한 학습데이터로 직접 학습해 사용하였습니다.
토크나이저의 학습은 학습데이터의 발화와 요약문을 mecab으로 1차분절한 후에 unigram 방법으로 학습해 사용하였습니다.

### 2. BART pretraining

저희는 모델 아키텍쳐로는 BART를 사용했습니다.
외부 데이터를 사용할 수 없기 때문에 학습 데이터셋의 대화 텍스트로 사전학습을 수행했으며 노이즈는 Text infilling과 Sentence Permutation을 적용했습니다.
Sentence Permutation은 턴 단위로 수행하였습니다.

### 3. Dialogue Summarization finetune (R3F)

사전학습 후에는 Dialogue Summarization task를 학습시켰습니다.
이때 Abstract Summarization에서 좋은 효과를 보이는 R3F 기법을 적용하였습니다.

### 4. Dialogue Summarization finetune (RL)

마지막으로 학습의 목표를 대회의 평가지표인 ROUGE-L F1 score와 align시키기 위해서 고전적인 RL을 적용했습니다.
target metric은 모델이 생성한 요약문과 실제 요약문간의 mecab분절 기준 ROUGE-L F1 score를 사용하였습니다.

## References

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)
- [Better Fine-Tuning by Reducing Representational Collapse](https://arxiv.org/abs/2008.03156v1)
- [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)
