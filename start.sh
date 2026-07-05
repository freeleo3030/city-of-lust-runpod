#!/bin/bash

VOLUME_DIR="/runpod-volume"
MODEL_DIR="/comfyui/models/checkpoints"
CN_DIR="/comfyui/models/controlnet"

# 체크포인트 모델 링크
mkdir -p "$MODEL_DIR"
if [ -f "${VOLUME_DIR}/chilloutmix.safetensors" ]; then
    ln -sf "${VOLUME_DIR}/chilloutmix.safetensors" "${MODEL_DIR}/chilloutmix.safetensors"
fi

# ControlNet 모델 링크
mkdir -p "$CN_DIR"
if [ -f "${VOLUME_DIR}/control_v11p_sd15_openpose.safetensors" ]; then
    ln -sf "${VOLUME_DIR}/control_v11p_sd15_openpose.safetensors" "${CN_DIR}/control_v11p_sd15_openpose.safetensors"
    echo "ControlNet model linked."
elif [ -f "${VOLUME_DIR}/control_v11p_sd15_openpose.pth" ]; then
    ln -sf "${VOLUME_DIR}/control_v11p_sd15_openpose.pth" "${CN_DIR}/control_v11p_sd15_openpose.pth"
    echo "ControlNet model (pth) linked."
else
    echo "WARNING: ControlNet model not found in volume. Pose generation will use txt2img fallback."
fi

echo "Starting handler..."
python -u /handler.py
