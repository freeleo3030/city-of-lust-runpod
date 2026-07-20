#!/bin/bash

VOLUME_DIR="/runpod-volume"
MODEL_DIR="/comfyui/models/checkpoints"
IPA_DIR="/comfyui/models/ipadapter"
CLIP_DIR="/comfyui/models/clip_vision"

# Juggernaut XL Ragnarok 체크포인트 링크
mkdir -p "$MODEL_DIR"
JUGGERNAUT_FILE="${VOLUME_DIR}/juggernaut_xl_ragnarok.safetensors"
MIN_SIZE=$((5 * 1024 * 1024 * 1024))  # 5GB 미만이면 손상된 파일로 간주
FILE_SIZE=0
if [ -f "$JUGGERNAUT_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$JUGGERNAUT_FILE" 2>/dev/null || echo 0)
fi

if [ -f "$JUGGERNAUT_FILE" ] && [ "$FILE_SIZE" -ge "$MIN_SIZE" ]; then
    ln -sf "$JUGGERNAUT_FILE" "${MODEL_DIR}/juggernaut_xl_ragnarok.safetensors"
    echo "Juggernaut XL Ragnarok linked (${FILE_SIZE} bytes)."
else
    if [ -f "$JUGGERNAUT_FILE" ]; then
        echo "Juggernaut XL: file exists but incomplete (${FILE_SIZE} bytes). Removing and re-downloading..."
        rm -f "$JUGGERNAUT_FILE"
    else
        echo "Juggernaut XL Ragnarok not found. Downloading from HuggingFace (~7GB)..."
    fi
    wget -q --show-progress \
        "https://huggingface.co/modelzpalace/juggernautXL_ragnarok/resolve/main/juggernautXL_ragnarokBy.safetensors" \
        -O "$JUGGERNAUT_FILE"
    ln -sf "$JUGGERNAUT_FILE" "${MODEL_DIR}/juggernaut_xl_ragnarok.safetensors"
    echo "Juggernaut XL Ragnarok downloaded and linked."
fi

# IP-Adapter SDXL Face 모델 링크
mkdir -p "$IPA_DIR" "$CLIP_DIR"
if [ -f "${VOLUME_DIR}/ip-adapter-plus-face_sdxl_vit-h.safetensors" ]; then
    ln -sf "${VOLUME_DIR}/ip-adapter-plus-face_sdxl_vit-h.safetensors" "${IPA_DIR}/ip-adapter-plus-face_sdxl_vit-h.safetensors"
    echo "IP-Adapter SDXL linked."
elif [ -f "${VOLUME_DIR}/ip-adapter-plus-face_sdxl_vit-h.bin" ]; then
    ln -sf "${VOLUME_DIR}/ip-adapter-plus-face_sdxl_vit-h.bin" "${IPA_DIR}/ip-adapter-plus-face_sdxl_vit-h.bin"
    echo "IP-Adapter SDXL linked (bin)."
else
    echo "WARNING: IP-Adapter SDXL model not found. Downloading..."
    wget -q --show-progress \
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" \
        -O "${VOLUME_DIR}/ip-adapter-plus-face_sdxl_vit-h.safetensors"
    ln -sf "${VOLUME_DIR}/ip-adapter-plus-face_sdxl_vit-h.safetensors" "${IPA_DIR}/ip-adapter-plus-face_sdxl_vit-h.safetensors"
    echo "IP-Adapter SDXL downloaded and linked."
fi

# CLIP ViT-H-14 (SDXL IPAdapter에서도 동일 모델 사용)
CLIP_H_PATH="${VOLUME_DIR}/clip-vit-h-14.safetensors"
if [ ! -f "${CLIP_H_PATH}" ]; then
    echo "Downloading CLIP ViT-H-14 (~2.5GB)..."
    wget -q --show-progress \
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
        -O "${CLIP_H_PATH}"
    echo "CLIP ViT-H-14 download done."
fi
ln -sf "${CLIP_H_PATH}" "${CLIP_DIR}/clip-vit-h-14.safetensors"
echo "CLIP Vision (ViT-H-14) linked."

echo "Starting handler..."
python -u /handler.py
