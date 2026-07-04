import runpod
import base64
import os
from io import BytesIO

print("Importing torch...")
import torch
print("Importing diffusers...")
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
print("Importing PIL...")
from PIL import Image
print("All imports done!")

VOLUME_DIR = "/runpod-volume"
MODEL_PATH = f"{VOLUME_DIR}/chilloutmix.safetensors"

pipe_txt2img = None
pipe_img2img = None

def load_pipeline():
    global pipe_txt2img, pipe_img2img

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    print(f"Loading model from {MODEL_PATH}...")
    pipe_txt2img = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    pipe_img2img = StableDiffusionImg2ImgPipeline(
        vae=pipe_txt2img.vae,
        text_encoder=pipe_txt2img.text_encoder,
        tokenizer=pipe_txt2img.tokenizer,
        unet=pipe_txt2img.unet,
        scheduler=pipe_txt2img.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    print("Model loaded!")

def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("RGB")

def handler(job):
    global pipe_txt2img, pipe_img2img
    if pipe_txt2img is None:
        load_pipeline()

    inp = job["input"]
    mode = inp.get("mode", "txt2img")
    prompt = inp.get("prompt", "")
    negative_prompt = inp.get("negative_prompt", "")
    width = inp.get("width", 512)
    height = inp.get("height", 768)
    steps = inp.get("steps", 28)
    cfg_scale = inp.get("cfg_scale", 7)
    seed = inp.get("seed", -1)
    denoising_strength = inp.get("denoising_strength", 0.75)
    init_image_b64 = inp.get("init_image", None)

    generator = torch.Generator("cuda")
    if seed != -1:
        generator.manual_seed(seed)

    try:
        if mode == "img2img" and init_image_b64:
            init_image = b64_to_image(init_image_b64).resize((width, height))
            result = pipe_img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=denoising_strength,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )
        else:
            result = pipe_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )
        return {"image": image_to_b64(result.images[0]), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})
