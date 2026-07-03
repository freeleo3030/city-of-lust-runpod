import runpod
import requests
import base64
import subprocess
import time
import os

A1111_URL = "http://127.0.0.1:7860"

def start_a1111():
    subprocess.Popen([
        "python", "/stable-diffusion-webui/webui.py",
        "--api", "--nowebui", "--skip-torch-cuda-test",
        "--no-half-vae", "--xformers",
        "--ckpt-dir", "/stable-diffusion-webui/models/Stable-diffusion",
    ])
    # A1111 시작 대기
    for _ in range(60):
        try:
            r = requests.get(f"{A1111_URL}/sdapi/v1/sd-models", timeout=2)
            if r.status_code == 200:
                print("A1111 ready")
                return True
        except:
            time.sleep(3)
    raise RuntimeError("A1111 failed to start")

def txt2img(payload: dict) -> str:
    r = requests.post(f"{A1111_URL}/sdapi/v1/txt2img", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["images"][0]  # base64

def img2img(payload: dict) -> str:
    r = requests.post(f"{A1111_URL}/sdapi/v1/img2img", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["images"][0]  # base64

def handler(job):
    input_data = job["input"]
    mode = input_data.get("mode", "txt2img")  # txt2img | img2img
    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "")
    width = input_data.get("width", 512)
    height = input_data.get("height", 768)
    steps = input_data.get("steps", 28)
    cfg_scale = input_data.get("cfg_scale", 7)
    seed = input_data.get("seed", -1)
    denoising_strength = input_data.get("denoising_strength", 0.75)
    init_image = input_data.get("init_image", None)  # base64, img2img용
    controlnet_image = input_data.get("controlnet_image", None)  # base64, 포즈용

    base_payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sampler_name": "DPM++ 2M Karras",
        "override_settings": {
            "sd_model_checkpoint": "chilloutmix_NiPrunedFp32Fix.safetensors"
        },
        "alwayson_scripts": {
            "ADetailer": {
                "args": [True, {"ad_model": "face_yolov8n.pt"}]
            }
        }
    }

    # ControlNet 포즈 이미지 있으면 추가
    if controlnet_image:
        base_payload["alwayson_scripts"]["ControlNet"] = {
            "args": [{
                "input_image": controlnet_image,
                "module": "openpose",
                "model": "control_sd15_openpose",
                "weight": 0.8,
                "enabled": True
            }]
        }

    try:
        if mode == "img2img" and init_image:
            payload = {
                **base_payload,
                "init_images": [init_image],
                "denoising_strength": denoising_strength
            }
            image_b64 = img2img(payload)
        else:
            image_b64 = txt2img(base_payload)

        return {"image": image_b64, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "failed"}


# A1111 시작
start_a1111()
runpod.serverless.start({"handler": handler})
