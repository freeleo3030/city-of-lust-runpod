#!/bin/bash

VOLUME_DIR="/runpod-volume"
MODEL_PATH="${VOLUME_DIR}/chilloutmix.safetensors"
CONTROLNET_PATH="${VOLUME_DIR}/control_sd15_openpose.pth"

mkdir -p /stable-diffusion-webui/models/Stable-diffusion
mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/models

# ChilloutMix 없으면 Volume에 다운로드
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading ChilloutMix to volume..."
    wget -q -O "$MODEL_PATH" \
        "https://huggingface.co/TASUKU2023/Chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors"
fi
ln -sf "$MODEL_PATH" /stable-diffusion-webui/models/Stable-diffusion/chilloutmix.safetensors

# ControlNet 없으면 Volume에 다운로드
if [ ! -f "$CONTROLNET_PATH" ]; then
    echo "Downloading ControlNet OpenPose to volume..."
    wget -q -O "$CONTROLNET_PATH" \
        "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth"
fi
ln -sf "$CONTROLNET_PATH" /stable-diffusion-webui/extensions/sd-webui-controlnet/models/control_sd15_openpose.pth

# A1111 필요한 repo 초기화 (최초 1회)
cd /stable-diffusion-webui
if [ ! -d "repositories/stable-diffusion-stability-ai" ]; then
    echo "Initializing A1111 repositories..."
    python launch.py --skip-torch-cuda-test --exit
fi

echo "Starting handler..."
python -u /handler.py
