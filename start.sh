#!/bin/bash

MODEL_PATH="/stable-diffusion-webui/models/Stable-diffusion/chilloutmix.safetensors"
CONTROLNET_PATH="/stable-diffusion-webui/extensions/sd-webui-controlnet/models/control_sd15_openpose.pth"

# ChilloutMix 모델 없으면 다운로드
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading ChilloutMix..."
    mkdir -p /stable-diffusion-webui/models/Stable-diffusion
    wget -q -O "$MODEL_PATH" \
        "https://huggingface.co/TASUKU2023/Chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors"
fi

# ControlNet 모델 없으면 다운로드
if [ ! -f "$CONTROLNET_PATH" ]; then
    echo "Downloading ControlNet OpenPose..."
    mkdir -p /stable-diffusion-webui/extensions/sd-webui-controlnet/models
    wget -q -O "$CONTROLNET_PATH" \
        "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth"
fi

echo "Starting handler..."
python -u /handler.py
