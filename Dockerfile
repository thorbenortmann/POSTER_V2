FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.9 \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

CMD ["/bin/bash"]

# This image let's you create an isolated dev environment for POSTER_V2
# docker build -t pv2-train .
# docker run --rm -it --gpus all --shm-size=8g -v .:/app/POSTER_V2 pv2-train
