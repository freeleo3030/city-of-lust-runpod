#!/bin/bash

VOLUME_DIR="/runpod-volume"
MODEL_PATH="${VOLUME_DIR}/chilloutmix.safetensors"

# ChilloutMix 없으면 Volume에 다운로드
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading ChilloutMix to volume..."
    wget --progress=dot:giga -O "$MODEL_PATH" \
        "https://huggingface.co/TASUKU2023/Chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors" 2>&1
    if [ $? -ne 0 ]; then
        echo "Download failed! Trying mirror..."
        wget --progress=dot:giga -O "$MODEL_PATH" \
            "https://huggingface.co/TASUKU2023/Chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors?download=true" 2>&1
    fi
    echo "Download done, file size: $(du -sh $MODEL_PATH 2>/dev/null)"
fi

echo "Starting handler..."
python -u /handler.py
