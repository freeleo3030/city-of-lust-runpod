#!/bin/bash

VOLUME_DIR="/runpod-volume"
MODEL_PATH="${VOLUME_DIR}/chilloutmix.safetensors"

# ChilloutMix 없으면 Volume에 다운로드
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading ChilloutMix to volume..."
    wget -q -O "$MODEL_PATH" \
        "https://huggingface.co/TASUKU2023/Chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors"
    echo "Download complete!"
fi

echo "Starting handler..."
python -u /handler.py
