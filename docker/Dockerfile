FROM cosmoquester/mecab AS mecab

FROM python:3.8

COPY --from=mecab /usr/local/etc/mecabrc /usr/local/etc/mecabrc
COPY --from=mecab /usr/local/lib/* /usr/local/lib/
COPY --from=mecab /usr/local/bin/* /usr/local/bin/
COPY --from=mecab /opt/mecab-ko-dic /opt/mecab-ko-dic
COPY --from=mecab /usr/local/libexec/mecab /usr/local/libexec/mecab
RUN ldconfig

WORKDIR /opt/mecab-ko-dic

COPY ./docker/dialogue-summary-specials.csv ./user-dic

RUN ./tools/add-userdic.sh

RUN make install

RUN apt update && apt install git-lfs

RUN git lfs install

COPY requirements.txt .

RUN pip install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install -r requirements.txt

WORKDIR /app

COPY summarizer summarizer
COPY run run

ENTRYPOINT ["python3", "-m", "run"]
