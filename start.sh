#!/bin/bash

VOLUME_DIR="/runpod-volume"
MODEL_DIR="/comfyui/models/checkpoints"

# Volume의 모델을 ComfyUI 모델 폴더로 링크
mkdir -p "$MODEL_DIR"
if [ -f "${VOLUME_DIR}/chilloutmix.safetensors" ]; then
    ln -sf "${VOLUME_DIR}/chilloutmix.safetensors" "${MODEL_DIR}/chilloutmix.safetensors"
    echo "Model linked from volume."
else
    echo "WARNING: Model not found in volume!"
fi

# ComfyUI 서버 백그라운드 실행
echo "Starting ComfyUI server..."
python /comfyui/main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch &
COMFYUI_PID=$!

# ComfyUI 준비될 때까지 대기
echo "Waiting for ComfyUI to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    sleep 2
done

echo "Starting handler..."
python -u /handler.py
