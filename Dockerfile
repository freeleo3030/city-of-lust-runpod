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

# ControlNet 설치
RUN git clone https://github.com/Mikubill/sd-webui-controlnet extensions/sd-webui-controlnet

# ADetailer 설치
RUN git clone https://github.com/Bing-su/adetailer extensions/adetailer

# A1111 repo 초기화 (빌드 시점에 미리 — 런타임에 GitHub 접근 불가)
RUN python launch.py --skip-torch-cuda-test --exit || true

# RunPod 핸들러
RUN pip install runpod requests
COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
