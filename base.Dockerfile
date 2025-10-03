# Base with CUDA 12.8 (matches torch cu128)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS runtime

# Clean any third-party apt sources (avoid key expiry)
RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Build for common GPUs incl. Blackwell (SM120)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Hugging Face faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /

# System deps
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    git wget curl bash libgl1 software-properties-common \
    openssh-server nginx rsync ffmpeg \
    build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev git-lfs && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install --yes --no-install-recommends python3.10-dev python3.10-venv && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Python & pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && rm -f get-pip.py

RUN pip install -U wheel setuptools packaging

# ---- PyTorch stack (CUDA 12.8) ----
# Matches what your logs show: torch 2.7.0+cu128, xformers 0.0.30
RUN pip install \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    pip install xformers==0.0.30

# (Optional) If you plan to use sageattention explicitly, it’s usually bundled by nodes,
# but installing the wheel here is harmless. Uncomment if needed:
# RUN pip install sageattention

# ---- Project bits kept from original base ----
WORKDIR /
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git
WORKDIR /InfiniteTalk

# Core python deps
RUN pip install "misaki[en]" ninja psutil packaging
# NOTE: Removed flash_attn to prevent SM90-only kernel issues.
# RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# Project requirements
RUN pip install -r requirements.txt

# Extra runtime deps
RUN pip install librosa ffmpeg onnx sageattention
RUN pip uninstall -y transformers && pip install transformers==4.48.2

# Serverless / infra helpers + HF fast transfer
RUN pip install runpod websocket-client && \
    pip install -U "huggingface_hub[hf_transfer]"

# ---- Notes ----
# docker build -t krypton8/multitalk-base:1.0 -f base.Dockerfile .
# docker push krypton8/multitalk-base:1.0

# If you previously published 1.0, bump the tag to 1.1 after this change.
