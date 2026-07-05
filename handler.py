import runpod
import base64
import random
import sys
import os

print("handler.py starting...", flush=True)

# ComfyUI를 라이브러리로 직접 사용
sys.path.insert(0, '/comfyui')

MODEL_PATH = "/comfyui/models/checkpoints/chilloutmix.safetensors"

# 모델을 미리 로드 (cold start 시 한 번만)
loaded_model = None
loaded_clip = None
loaded_vae = None

def load_model():
    global loaded_model, loaded_clip, loaded_vae
    if loaded_model is not None:
        return

    print("Loading ComfyUI modules...", flush=True)
    import comfy.model_management as model_management
    from nodes import CheckpointLoaderSimple

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    print(f"Loading model from {MODEL_PATH}...", flush=True)
    loader = CheckpointLoaderSimple()
    loaded_model, loaded_clip, loaded_vae = loader.load_checkpoint("chilloutmix.safetensors")
    print("Model loaded!", flush=True)

def generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed):
    import torch
    import comfy.model_management as model_management
    from nodes import (
        CLIPTextEncode, KSampler, VAEDecode,
        EmptyLatentImage
    )

    # 텍스트 인코딩
    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    # 빈 레이턴트 생성
    latent_creator = EmptyLatentImage()
    latent = latent_creator.generate(width, height, 1)[0]

    # 샘플링
    sampler = KSampler()
    sampled = sampler.sample(
        loaded_model, seed, steps, cfg_scale,
        "euler_ancestral", "karras",
        positive, negative, latent, denoise=1.0
    )[0]

    # VAE 디코딩
    decoder = VAEDecode()
    image = decoder.decode(loaded_vae, sampled)[0]

    return image

def handler(job):
    try:
        inp = job["input"]
        prompt = inp.get("prompt", "")
        negative_prompt = inp.get("negative_prompt", "")
        width = inp.get("width", 512)
        height = inp.get("height", 768)
        steps = inp.get("steps", 25)
        cfg_scale = inp.get("cfg_scale", 7)
        seed = inp.get("seed", -1)

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        load_model()

        print(f"Generating image: {width}x{height}, steps={steps}, seed={seed}", flush=True)
        image_tensor = generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed)

        # 이미지 텐서 → PIL → base64
        from PIL import Image
        import numpy as np
        i = 255. * image_tensor[0].detach().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"image": image_b64, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return {"error": str(e), "status": "failed"}

print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
