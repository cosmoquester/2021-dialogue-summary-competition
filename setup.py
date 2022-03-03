from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="2021-dialogue-summary-competition",
    version="1.1.0",
    description="[2021 훈민정음 한국어 음성•자연어 인공지능 경진대회] 대화요약 부문 알라꿍달라꿍 팀의 대화요약 학습 및 추론 코드를 공유하기 위한 레포입니다.",
    long_description=long_description,
    python_requires=">=3.7",
    install_requires=["torch", "transformers", "pytorch-lightning", "wandb"],
    extras_require={"rl": ["mecab-python3", "rouge-score"], "tokenizer": ["sentencepiece", "httpimport"]},
    url="https://github.com/cosmoquester/2021-dialogue-summary-competition.git",
    author="Park Sangjun",
    keywords=["nlp", "dialogue", "summarization", "huggingface-transformers", "pytorch-lightning"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    packages=find_packages(exclude=["tests"]),
)
