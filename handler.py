import runpod
import base64
import random
import sys
import os

print("handler.py starting...", flush=True)

sys.path.insert(0, '/comfyui')

MODEL_PATH = "/comfyui/models/checkpoints/chilloutmix.safetensors"
CN_PATH = "/comfyui/models/controlnet/control_v11p_sd15_openpose.safetensors"
CN_PATH_FALLBACK = "/comfyui/models/controlnet/control_v11p_sd15_openpose.pth"
IPA_PATH = "/comfyui/models/ipadapter/ip-adapter-plus-face_sd15.bin"
CLIP_PATH = "/comfyui/models/clip_vision/clip-vit-large-patch14.bin"

loaded_ipadapter = None
loaded_clip_vision = None

loaded_model = None
loaded_clip = None
loaded_vae = None
loaded_controlnet = None


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


def load_controlnet():
    global loaded_controlnet
    if loaded_controlnet is not None:
        return True
    if os.path.exists(CN_PATH):
        cn_filename = "control_v11p_sd15_openpose.safetensors"
    elif os.path.exists(CN_PATH_FALLBACK):
        cn_filename = "control_v11p_sd15_openpose.pth"
    else:
        print("ControlNet model not found, skipping.", flush=True)
        return False
    from nodes import ControlNetLoader
    print(f"Loading ControlNet model ({cn_filename})...", flush=True)
    loader = ControlNetLoader()
    loaded_controlnet = loader.load_controlnet(cn_filename)[0]
    print("ControlNet loaded!", flush=True)
    return True


def load_ipadapter():
    global loaded_ipadapter, loaded_clip_vision
    if loaded_ipadapter is not None:
        return True
    if not os.path.exists(IPA_PATH) or not os.path.exists(CLIP_PATH):
        print("IP-Adapter or CLIP Vision model not found, skipping.", flush=True)
        return False
    try:
        from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterModelLoader
        from nodes import CLIPVisionLoader
        print("Loading IP-Adapter model...", flush=True)
        loader = IPAdapterModelLoader()
        loaded_ipadapter = loader.load_ipadapter_model("ip-adapter-plus-face_sd15.bin")[0]
        clip_loader = CLIPVisionLoader()
        loaded_clip_vision = clip_loader.load_clip("clip-vit-large-patch14.bin")[0]
        print("IP-Adapter loaded!", flush=True)
        return True
    except Exception as e:
        print(f"IP-Adapter load failed: {e}", flush=True)
        return False


def ipadapter_img2img(prompt, negative_prompt, pose_image_b64, face_image_b64, width, height, steps, cfg_scale, seed, ipa_strength=0.7, denoise=0.85):
    import torch
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from nodes import CLIPTextEncode, KSampler, VAEDecode, VAEEncode
    from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapter

    def b64_to_tensor(b64):
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB").resize((width, height), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    pose_tensor = b64_to_tensor(pose_image_b64)
    face_tensor = b64_to_tensor(face_image_b64)

    # pose 이미지를 init_image로 인코딩
    vae_encoder = VAEEncode()
    latent = vae_encoder.encode(loaded_vae, pose_tensor)[0]

    # 텍스트 인코딩
    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative_cond = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    # IP-Adapter 적용 (얼굴 이미지 기반) - 단순 IPAdapter 사용
    ipa_node = IPAdapter(loaded_ipadapter)
    result = ipa_node.apply_ipadapter(
        loaded_model, loaded_clip_vision,
        face_tensor, ipa_strength, 0, 1
    )
    model_with_ipa = result[0]

    sampler = KSampler()
    sampled = sampler.sample(
        model_with_ipa, seed, steps, cfg_scale,
        "euler_ancestral", "karras",
        positive, negative_cond, latent, denoise=denoise
    )[0]

    decoder = VAEDecode()
    return decoder.decode(loaded_vae, sampled)[0]


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

    img_bytes = base64.b64decode(init_image_b64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    pil_img = pil_img.resize((width, height), Image.LANCZOS)

    np_img = np.array(pil_img).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np_img).unsqueeze(0)

    vae_encoder = VAEEncode()
    latent = vae_encoder.encode(loaded_vae, image_tensor)[0]

    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    sampler = KSampler()
    sampled = sampler.sample(
        loaded_model, seed, steps, cfg_scale,
        "euler_ancestral", "karras",
        positive, negative, latent, denoise=denoising_strength
    )[0]

    decoder = VAEDecode()
    return decoder.decode(loaded_vae, sampled)[0]


def controlnet_img(prompt, negative_prompt, pose_image_b64, width, height, steps, cfg_scale, seed, cn_strength=1.0):
    import torch
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage, ControlNetApply

    # skeleton 이미지 로드
    img_bytes = base64.b64decode(pose_image_b64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((width, height), Image.LANCZOS)
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    pose_tensor = torch.from_numpy(np_img).unsqueeze(0)

    # 텍스트 인코딩
    clip_encoder = CLIPTextEncode()
    positive_base = clip_encoder.encode(loaded_clip, prompt)[0]
    negative_cond = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    # ControlNet 적용
    cn_apply = ControlNetApply()
    positive = cn_apply.apply_controlnet(positive_base, loaded_controlnet, pose_tensor, cn_strength)[0]

    # 생성
    latent_creator = EmptyLatentImage()
    latent = latent_creator.generate(width, height, 1)[0]

    sampler = KSampler()
    sampled = sampler.sample(
        loaded_model, seed, steps, cfg_scale,
        "euler_ancestral", "karras",
        positive, negative_cond, latent, denoise=1.0
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
        mode = inp.get("mode", "txt2img")  # "txt2img" | "img2img" | "controlnet"
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

        if mode == "ipadapter":
            pose_image = inp.get("pose_image", "")
            face_image = inp.get("face_image", "")
            ipa_strength = float(inp.get("ipa_strength", 0.7))
            denoise = float(inp.get("denoising_strength", 0.85))
            if not pose_image or not face_image:
                raise ValueError("ipadapter mode requires pose_image and face_image (base64)")
            ipa_ok = load_ipadapter()
            if not ipa_ok:
                # fallback: img2img without face
                print("Falling back to img2img (no IP-Adapter)", flush=True)
                image_tensor = img2img(prompt, negative_prompt, pose_image, denoise, width, height, steps, cfg_scale, seed)
            else:
                image_tensor = ipadapter_img2img(
                    prompt, negative_prompt, pose_image, face_image,
                    width, height, steps, cfg_scale, seed, ipa_strength, denoise
                )
        elif mode == "controlnet":
            pose_image = inp.get("pose_image", "")
            cn_strength = float(inp.get("cn_strength", 1.0))
            if not pose_image:
                raise ValueError("controlnet mode requires pose_image (base64)")
            cn_ok = load_controlnet()
            if not cn_ok:
                # ControlNet 없으면 txt2img fallback
                print("Falling back to txt2img (no ControlNet model)", flush=True)
                image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
            else:
                image_tensor = controlnet_img(
                    prompt, negative_prompt, pose_image,
                    width, height, steps, cfg_scale, seed, cn_strength
                )
        elif mode == "img2img":
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
