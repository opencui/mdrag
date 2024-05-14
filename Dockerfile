FROM ubuntu:22.04
ENV  PKG    wget curl gnupg2 git python3-pip python3-venv
RUN  apt-get update && apt-get install -y $PKG
RUN  ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /data/
COPY . .
RUN  python3 -m pip install -U pip && python3 -m pip install --no-cache-dir -r requirements.txt && python3 rag-serve.py
