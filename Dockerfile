FROM runpod/stable-diffusion-webui:latest

# 환경변수
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ChilloutMix 모델 다운로드
RUN wget -q -O /stable-diffusion-webui/models/Stable-diffusion/chilloutmix.safetensors \
    "https://huggingface.co/TASUKU2023/Chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors"

# ControlNet 익스텐션 설치
RUN git clone https://github.com/Mikubill/sd-webui-controlnet \
    /stable-diffusion-webui/extensions/sd-webui-controlnet

# ControlNet OpenPose 모델 다운로드
RUN mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/models && \
    wget -q -O /stable-diffusion-webui/extensions/sd-webui-controlnet/models/control_openpose.pth \
    "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth"

# ADetailer 익스텐션 설치 (얼굴 자동 보정)
RUN git clone https://github.com/Bing-su/adetailer \
    /stable-diffusion-webui/extensions/adetailer

# RunPod 핸들러
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
