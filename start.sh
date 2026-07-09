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
    echo "WARNING: ControlNet model not found in volume."
fi

# IP-Adapter 모델 링크
IPA_DIR="/comfyui/models/ipadapter"
CLIP_DIR="/comfyui/models/clip_vision"
mkdir -p "$IPA_DIR" "$CLIP_DIR"
if [ -f "${VOLUME_DIR}/ip-adapter-plus-face_sd15.safetensors" ]; then
    ln -sf "${VOLUME_DIR}/ip-adapter-plus-face_sd15.safetensors" "${IPA_DIR}/ip-adapter-plus-face_sd15.safetensors"
    echo "IP-Adapter model linked."
elif [ -f "${VOLUME_DIR}/ip-adapter-plus-face_sd15.bin" ]; then
    ln -sf "${VOLUME_DIR}/ip-adapter-plus-face_sd15.bin" "${IPA_DIR}/ip-adapter-plus-face_sd15.bin"
    echo "IP-Adapter model linked (bin)."
fi
if [ -f "${VOLUME_DIR}/clip-vit-large-patch14.bin" ]; then
    ln -sf "${VOLUME_DIR}/clip-vit-large-patch14.bin" "${CLIP_DIR}/clip-vit-large-patch14.bin"
    echo "CLIP Vision model linked."
fi

echo "Starting handler..."
python -u /handler.py
