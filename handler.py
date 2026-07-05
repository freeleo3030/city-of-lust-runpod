import runpod
import base64
import json
import time
import random
import urllib.request

print("handler.py starting...", flush=True)

COMFYUI_URL = "http://127.0.0.1:8188"

def queue_prompt(prompt_workflow):
    data = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_image(filename, subfolder, folder_type):
    url = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
    with urllib.request.urlopen(url) as response:
        return response.read()

def build_workflow(prompt, negative_prompt, width, height, steps, cfg_scale, seed):
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "chilloutmix.safetensors"}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]}
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler_ancestral",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["4", 2]}
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "output", "images": ["9", 0]}
        }
    }

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

        workflow = build_workflow(prompt, negative_prompt, width, height, steps, cfg_scale, seed)

        print("Queueing prompt...", flush=True)
        result = queue_prompt(workflow)
        prompt_id = result["prompt_id"]
        print(f"Prompt ID: {prompt_id}", flush=True)

        # 완료될 때까지 대기 (최대 300초)
        for _ in range(300):
            history = get_history(prompt_id)
            if prompt_id in history:
                break
            time.sleep(1)

        # 이미지 가져오기
        outputs = history[prompt_id]["outputs"]
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                img_data = node_output["images"][0]
                image_bytes = get_image(img_data["filename"], img_data["subfolder"], img_data["type"])
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                return {"image": image_b64, "status": "success"}

        return {"error": "No image generated", "status": "failed"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return {"error": str(e), "status": "failed"}

print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
