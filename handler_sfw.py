import runpod
import base64
import random
import os
import gc
from io import BytesIO

print("handler_sfw.py starting... majicMIX Realistic SD1.5 SFW", flush=True)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

MODEL_ID = "Yntec/majicMIXRealistic"
MODEL_PATH = "/models/majicmix"

pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    print(f"Loading majicMIX Realistic from {MODEL_PATH}...", flush=True)
    if os.path.exists(MODEL_PATH):
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        print("Local model not found, downloading from HuggingFace...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.save_pretrained(MODEL_PATH)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("majicMIX Realistic loaded!", flush=True)


def txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed):
    import torch
    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
    )
    img = result.images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    try:
        inp = job["input"]
        prompt = inp.get("prompt", "")
        negative_prompt = inp.get("negative_prompt", "")
        width = inp.get("width", 512)
        height = inp.get("height", 512)
        steps = inp.get("steps", 28)
        cfg_scale = inp.get("cfg_scale", 7)
        seed = inp.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        load_model()
        print(f"Generating {width}x{height}, steps={steps}, seed={seed}", flush=True)

        image_b64 = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)

        gc.collect()
        import torch
        torch.cuda.empty_cache()

        return {"image": image_b64, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return {"error": str(e), "status": "failed"}


print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
