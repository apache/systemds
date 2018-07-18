#1. Base image, contains OS with CUDA 8.0, cuDNN 5.1.10 prebuilt
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-setuptools

# 2. Install spark backend
RUN pip3 install pyspark

# 3. Install systemml
RUN pip3 install systemml

# 4. Install utilities
RUN pip3 install jupyter matplotlib numpy

# 5. Run Jupter notebook
RUN jupyter notebook
