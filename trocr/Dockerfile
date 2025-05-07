FROM nvidia/cuda:12.2.2-base-ubuntu20.04
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    locales \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /data/models
RUN mkdir -p /data/data/train_limited_50
RUN mkdir -p /data/data/val
COPY ./train_limited_50 /data/data/train_limited_50
COPY ./val /data/data/val
COPY ./models /data/models
COPY ./clearml.conf /
COPY ./requirements.txt /data/requirements.txt
RUN pip install -r /data/requirements.txt
RUN pip install boto3==1.9