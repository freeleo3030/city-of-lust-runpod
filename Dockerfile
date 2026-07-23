FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV COMFYUI_PATH=/comfyui

RUN apt-get update && apt-get install -y \
    wget git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI 설치
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
RUN pip install --no-cache-dir -r /comfyui/requirements.txt

# IP-Adapter custom nodes
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/requirements.txt || true

# runpod + requests + accelerate (SDXL 필수) + imageio (SVD mp4 변환)
RUN pip install --no-cache-dir runpod requests Pillow accelerate "imageio[ffmpeg]"

COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
