import runpod
import base64
import os
from io import BytesIO

print("handler.py starting...", flush=True)

VOLUME_DIR = "/runpod-volume"
MODEL_PATH = f"{VOLUME_DIR}/chilloutmix.safetensors"

pipe = None

def load_pipeline():
    global pipe
    print("Importing torch and diffusers...", flush=True)
    import torch
    from diffusers import StableDiffusionPipeline
    print(f"torch version: {torch.__version__}", flush=True)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    print(f"Loading model from {MODEL_PATH}...", flush=True)
    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    print("Model loaded!", flush=True)

def handler(job):
    global pipe
    try:
        if pipe is None:
            load_pipeline()

        from PIL import Image
        import torch

        inp = job["input"]
        prompt = inp.get("prompt", "")
        negative_prompt = inp.get("negative_prompt", "")
        width = inp.get("width", 512)
        height = inp.get("height", 768)
        steps = inp.get("steps", 20)
        cfg_scale = inp.get("cfg_scale", 7)
        seed = inp.get("seed", -1)

        generator = torch.Generator("cuda")
        if seed != -1:
            generator.manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
        )

        buf = BytesIO()
        result.images[0].save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image": image_b64, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return {"error": str(e), "status": "failed"}

print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
