FROM ubuntu:20.04
ENV  PKG    wget curl gnupg2 git python3-pip python3-venv
RUN  echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal           main restricted universe multiverse" >  /etc/apt/sources.list && \
     echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security  main restricted universe multiverse" >> /etc/apt/sources.list && \
     echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates   main restricted universe multiverse" >> /etc/apt/sources.list && \
     echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed  main restricted universe multiverse" >> /etc/apt/sources.list && \
     echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
     apt-get update && apt-get install -y $PKG
RUN  ln -s /usr/bin/python3 /usr/bin/python

WORKSPACE /data/
COPY . .
RUN  python3 -m pip install -U pip && python3 -m pip install --no-cache-dir -r requirements.txt
