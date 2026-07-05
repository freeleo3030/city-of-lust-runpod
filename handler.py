import runpod
import base64
import random
import sys
import os

print("handler.py starting...", flush=True)

sys.path.insert(0, '/comfyui')

MODEL_PATH = "/comfyui/models/checkpoints/chilloutmix.safetensors"

loaded_model = None
loaded_clip = None
loaded_vae = None

def load_model():
    global loaded_model, loaded_clip, loaded_vae
    if loaded_model is not None:
        return
    print("Loading ComfyUI modules...", flush=True)
    from nodes import CheckpointLoaderSimple
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    print(f"Loading model from {MODEL_PATH}...", flush=True)
    loader = CheckpointLoaderSimple()
    loaded_model, loaded_clip, loaded_vae = loader.load_checkpoint("chilloutmix.safetensors")
    print("Model loaded!", flush=True)


def txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed):
    from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage

    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    latent_creator = EmptyLatentImage()
    latent = latent_creator.generate(width, height, 1)[0]

    sampler = KSampler()
    sampled = sampler.sample(
        loaded_model, seed, steps, cfg_scale,
        "euler_ancestral", "karras",
        positive, negative, latent, denoise=1.0
    )[0]

    decoder = VAEDecode()
    return decoder.decode(loaded_vae, sampled)[0]


def img2img(prompt, negative_prompt, init_image_b64, denoising_strength, width, height, steps, cfg_scale, seed):
    import torch
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from nodes import CLIPTextEncode, KSampler, VAEDecode, VAEEncode

    # base64 → PIL → tensor
    img_bytes = base64.b64decode(init_image_b64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # 크기 맞추기 (width/height로 리사이즈)
    pil_img = pil_img.resize((width, height), Image.LANCZOS)

    # PIL → ComfyUI image tensor [B, H, W, C] 0~1 float
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np_img).unsqueeze(0)  # [1, H, W, 3]

    # VAE 인코딩 → latent
    vae_encoder = VAEEncode()
    latent = vae_encoder.encode(loaded_vae, image_tensor)[0]

    # 텍스트 인코딩
    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    # 샘플링 (denoise < 1.0 이면 img2img)
    sampler = KSampler()
    sampled = sampler.sample(
        loaded_model, seed, steps, cfg_scale,
        "euler_ancestral", "karras",
        positive, negative, latent, denoise=denoising_strength
    )[0]

    decoder = VAEDecode()
    return decoder.decode(loaded_vae, sampled)[0]


def tensor_to_b64(image_tensor):
    import numpy as np
    from PIL import Image
    from io import BytesIO
    i = 255.0 * image_tensor[0].detach().cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    try:
        inp = job["input"]
        mode = inp.get("mode", "txt2img")  # "txt2img" | "img2img"
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

        print(f"Mode={mode}, {width}x{height}, steps={steps}, seed={seed}", flush=True)

        if mode == "img2img":
            init_image = inp.get("init_image", "")
            denoising_strength = float(inp.get("denoising_strength", 0.75))
            if not init_image:
                raise ValueError("img2img requires init_image (base64)")
            image_tensor = img2img(
                prompt, negative_prompt, init_image,
                denoising_strength, width, height, steps, cfg_scale, seed
            )
        else:
            image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)

        image_b64 = tensor_to_b64(image_tensor)
        return {"image": image_b64, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return {"error": str(e), "status": "failed"}


print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
