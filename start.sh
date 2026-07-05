#!/bin/bash

VOLUME_DIR="/runpod-volume"
MODEL_DIR="/comfyui/models/checkpoints"

# Volume의 모델을 ComfyUI 모델 폴더로 링크
mkdir -p "$MODEL_DIR"
if [ -f "${VOLUME_DIR}/chilloutmix.safetensors" ]; then
    ln -sf "${VOLUME_DIR}/chilloutmix.safetensors" "${MODEL_DIR}/chilloutmix.safetensors"
fi

echo "Starting handler..."
python -u /handler.py
