FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# diffusers + 필요 패키지
RUN pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers \
    accelerate \
    safetensors \
    runpod \
    requests \
    Pillow

# diffusers import 검증
RUN python -c "import torch; import diffusers; print('imports ok, torch:', torch.__version__)"

COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
