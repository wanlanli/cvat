# ---- DL base image ----
FROM ubuntu:20.04

# non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# install basic deps
RUN apt-get update && apt-get install -y \
    curl git \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# # upgrade pip
RUN python3 -m pip install --upgrade pip --no-cache-dir

# install common DL libs (CPU version)
RUN pip3 install --no-cache-dir torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 \
    scikit-image \
    kornia \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git' \
    'git+https://github.com/cocodataset/panopticapi.git'


# set working dir for Nuclio
WORKDIR /opt/nuclio

# optional symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip