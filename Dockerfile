FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지
RUN apt-get update && apt-get install -y \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# A1111 설치
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui /stable-diffusion-webui

WORKDIR /stable-diffusion-webui

# A1111 의존성 설치
RUN pip install -r requirements.txt

# ChilloutMix 모델 다운로드
RUN mkdir -p models/Stable-diffusion && \
    wget -q -O models/Stable-diffusion/chilloutmix.safetensors \
    "https://huggingface.co/TASUKU2023/Chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors"

# ControlNet 설치
RUN git clone https://github.com/Mikubill/sd-webui-controlnet extensions/sd-webui-controlnet && \
    mkdir -p extensions/sd-webui-controlnet/models && \
    wget -q -O extensions/sd-webui-controlnet/models/control_sd15_openpose.pth \
    "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth"

# ADetailer 설치
RUN git clone https://github.com/Bing-su/adetailer extensions/adetailer

# RunPod 핸들러
RUN pip install runpod requests
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
