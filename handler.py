import runpod
import base64
import random
import sys
import os
import gc
import tracemalloc

print("handler.py starting... SDXL V1", flush=True)


os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

tracemalloc.start()

sys.path.insert(0, '/comfyui')

MODEL_PATH = "/comfyui/models/checkpoints/juggernaut_xl_ragnarok.safetensors"
IPA_PATH = "/comfyui/models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors"
IPA_PATH_FALLBACK = "/comfyui/models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.bin"
CLIP_PATH = "/comfyui/models/clip_vision/clip-vit-h-14.safetensors"
SVD_PATH = "/workspace/svd/svd_xt.safetensors"

loaded_ipadapter = None
loaded_clip_vision = None
loaded_model = None
loaded_clip = None
loaded_vae = None
loaded_svd_pipe = None


def _force_vram_free():
    import gc, torch
    try:
        import comfy.model_management as mm
        print(f"[VRAM] current_loaded_models: {len(mm.current_loaded_models)}개 (유지)", flush=True)
        try:
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[VRAM] soft_empty_cache error: {e}", flush=True)
    except Exception as e:
        print(f"[VRAM] mm state error: {e}", flush=True)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()
    log_vram("after _force_vram_free")


def load_model():
    global loaded_model, loaded_clip, loaded_vae
    if loaded_model is not None:
        return
    print("Loading ComfyUI modules...", flush=True)
    from nodes import CheckpointLoaderSimple

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model_size_gb = os.path.getsize(MODEL_PATH) / (1024 ** 3)
    print(f"Model file size: {model_size_gb:.2f} GB", flush=True)

    print(f"Loading Juggernaut XL Ragnarok...", flush=True)
    loader = CheckpointLoaderSimple()
    loaded_model, loaded_clip, loaded_vae = loader.load_checkpoint("juggernaut_xl_ragnarok.safetensors")
    print("Model loaded!", flush=True)
    log_vram("after load_model")
    load_ipadapter()


def load_ipadapter():
    global loaded_ipadapter, loaded_clip_vision
    if loaded_ipadapter is not None:
        return True
    ipa_filename = None
    if os.path.exists(IPA_PATH):
        ipa_filename = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    elif os.path.exists(IPA_PATH_FALLBACK):
        ipa_filename = "ip-adapter-plus-face_sdxl_vit-h.bin"
    if not ipa_filename or not os.path.exists(CLIP_PATH):
        print("IP-Adapter or CLIP Vision model not found, skipping.", flush=True)
        return False
    try:
        from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterModelLoader
        from nodes import CLIPVisionLoader
        print(f"Loading IP-Adapter SDXL ({ipa_filename})...", flush=True)
        loader = IPAdapterModelLoader()
        loaded_ipadapter = loader.load_ipadapter_model(ipa_filename)[0]
        clip_loader = CLIPVisionLoader()
        loaded_clip_vision = clip_loader.load_clip("clip-vit-h-14.safetensors")[0]
        print("IP-Adapter loaded!", flush=True)
        return True
    except Exception as e:
        print(f"IP-Adapter load failed: {e}", flush=True)
        return False


def ipadapter_txt2img(prompt, negative_prompt, face_image_b64, width, height, steps, cfg_scale, seed, ipa_strength=0.35):
    import torch
    import numpy as np
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024 / 1024 / 1024
    print(f"VRAM before IPA: {int(free/1024/1024)}MB free / {int(total/1024/1024)}MB total", flush=True)
    if free_gb < 2.0:
        print(f"VRAM too low ({free_gb:.1f}GB), skipping IPA → txt2img", flush=True)
        return txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)

    from PIL import Image
    from io import BytesIO
    from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage

    ipa_module = __import__(
        'custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus',
        fromlist=['IPAdapterAdvanced', 'IPAdapterPlus', 'IPAdapter']
    )
    IPAdapterClass = (
        getattr(ipa_module, 'IPAdapterAdvanced', None) or
        getattr(ipa_module, 'IPAdapterPlus', None) or
        getattr(ipa_module, 'IPAdapter', None)
    )
    if IPAdapterClass is None:
        raise ImportError("No usable IPAdapter class found")
    print(f"IPA txt2img using: {IPAdapterClass.__name__}", flush=True)

    if ',' in face_image_b64:
        face_image_b64 = face_image_b64.split(',', 1)[1]
    face_b = face_image_b64.strip().encode('ascii', errors='ignore').decode('ascii')
    face_b += '=' * (-len(face_b) % 4)
    face_img = Image.open(BytesIO(base64.b64decode(face_b))).convert("RGB").resize((224, 224), Image.LANCZOS)
    face_arr = np.array(face_img).astype(np.float32) / 255.0
    face_tensor = torch.from_numpy(face_arr).unsqueeze(0)

    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative_cond = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    latent_creator = EmptyLatentImage()
    latent = latent_creator.generate(width, height, 1)[0]

    ipa_node = IPAdapterClass()
    result = ipa_node.apply_ipadapter(
        model=loaded_model, ipadapter=loaded_ipadapter,
        clip_vision=loaded_clip_vision, image=face_tensor,
        weight=ipa_strength, weight_type="original",
        start_at=0.0, end_at=1.0
    )
    model_with_ipa = result[0]
    del result

    try:
        sampler = KSampler()
        sampled = sampler.sample(
            model_with_ipa, seed, steps, cfg_scale,
            "euler_ancestral", "karras",
            positive, negative_cond, latent, denoise=1.0
        )[0]

        decoder = VAEDecode()
        result_image = decoder.decode(loaded_vae, sampled)[0]
        return result_image
    finally:
        try:
            import comfy.model_management as mm
            mm.current_loaded_models[:] = [
                lm for lm in mm.current_loaded_models
                if getattr(lm, 'model', None) is not model_with_ipa
                and lm is not model_with_ipa
            ]
        except Exception:
            pass
        try:
            if hasattr(model_with_ipa, 'patches') and isinstance(model_with_ipa.patches, dict):
                model_with_ipa.patches.clear()
        except Exception:
            pass
        del model_with_ipa, positive, negative_cond, latent, ipa_node
        try:
            del sampled
        except Exception:
            pass
        del face_tensor
        gc.collect()
        _force_vram_free()


def unload_main_model():
    global loaded_model, loaded_clip, loaded_vae, loaded_ipadapter, loaded_clip_vision
    import torch, gc
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    loaded_model = loaded_clip = loaded_vae = None
    loaded_ipadapter = loaded_clip_vision = None
    gc.collect()
    torch.cuda.empty_cache()
    print("Main model unloaded", flush=True)


def load_svd():
    global loaded_svd_pipe
    if loaded_svd_pipe is not None:
        return True
    if not os.path.exists(SVD_PATH):
        print(f"SVD model not found: {SVD_PATH}", flush=True)
        return False
    try:
        import torch
        from diffusers import StableVideoDiffusionPipeline
        loaded_svd_pipe = StableVideoDiffusionPipeline.from_single_file(
            SVD_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        loaded_svd_pipe = loaded_svd_pipe.to("cuda")
        loaded_svd_pipe.enable_model_cpu_offload()
        print("SVD model loaded!", flush=True)
        log_vram("after load_svd")
        return True
    except Exception as e:
        print(f"SVD load failed: {e}", flush=True)
        return False


def svd_generate(init_image_b64, num_frames=14, motion_bucket_id=127, fps=7, augmentation_level=0.0, steps=20, seed=42):
    import torch
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import imageio
    import tempfile

    # 입력 이미지 디코딩 및 SVD 권장 해상도(1024x576)로 리사이즈
    if ',' in init_image_b64:
        init_image_b64 = init_image_b64.split(',', 1)[1]
    img_b = init_image_b64.strip().encode('ascii', errors='ignore').decode('ascii')
    img_b += '=' * (-len(img_b) % 4)
    pil_img = Image.open(BytesIO(base64.b64decode(img_b))).convert("RGB").resize((1024, 576), Image.LANCZOS)

    generator = torch.manual_seed(seed)
    frames = loaded_svd_pipe(
        pil_img,
        num_frames=num_frames,
        num_inference_steps=steps,
        motion_bucket_id=motion_bucket_id,
        fps=fps,
        noise_aug_strength=augmentation_level,
        decode_chunk_size=4,
        generator=generator,
    ).frames[0]

    # 프레임 → mp4
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.close()
    writer = imageio.get_writer(tmp.name, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

    with open(tmp.name, 'rb') as f:
        mp4_b64 = base64.b64encode(f.read()).decode('utf-8')
    os.unlink(tmp.name)

    gc.collect()
    torch.cuda.empty_cache()
    return mp4_b64


def img2img(prompt, negative_prompt, init_image_b64, width, height, steps, cfg_scale, seed, denoise=0.5):
    import torch
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from nodes import CLIPTextEncode, KSampler, VAEDecode, VAEEncode

    if ',' in init_image_b64:
        init_image_b64 = init_image_b64.split(',', 1)[1]
    img_b = init_image_b64.strip().encode('ascii', errors='ignore').decode('ascii')
    img_b += '=' * (-len(img_b) % 4)
    pil_img = Image.open(BytesIO(base64.b64decode(img_b))).convert("RGB").resize((width, height), Image.LANCZOS)
    img_arr = np.array(pil_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative_cond = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    encoder = VAEEncode()
    latent = encoder.encode(loaded_vae, img_tensor)[0]

    sampler = KSampler()
    sampled = sampler.sample(
        loaded_model, seed, steps, cfg_scale,
        "euler_ancestral", "karras",
        positive, negative_cond, latent, denoise=denoise
    )[0]

    decoder = VAEDecode()
    result = decoder.decode(loaded_vae, sampled)[0]
    del positive, negative_cond, latent, sampled, img_tensor
    gc.collect()
    return result


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
    result = decoder.decode(loaded_vae, sampled)[0]
    del positive, negative, latent, sampled
    gc.collect()
    return result


def tensor_to_b64(image_tensor):
    import numpy as np
    from PIL import Image
    from io import BytesIO
    i = 255.0 * image_tensor[0].detach().cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def log_vram(label=""):
    try:
        import torch, psutil
        free, total = torch.cuda.mem_get_info()
        proc = psutil.Process(os.getpid())
        ram_gb = proc.memory_info().rss / 1024**3
        print(f"[VRAM]{' ' + label if label else ''}: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total | RAM: {ram_gb:.2f}GB", flush=True)
    except Exception:
        pass


def handler(job):
    try:
        inp = job["input"]
        mode = inp.get("mode", "txt2img")
        prompt = inp.get("prompt", "")
        negative_prompt = inp.get("negative_prompt", "")
        width = inp.get("width", 832)
        height = inp.get("height", 1216)
        steps = inp.get("steps", 30)
        cfg_scale = inp.get("cfg_scale", 7)
        seed = inp.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        load_model()
        log_vram("before generation")
        print(f"Mode={mode}, {width}x{height}, steps={steps}, seed={seed}", flush=True)

        if mode == "svd":
            init_image = inp.get("init_image", "")
            if not init_image:
                raise ValueError("svd mode requires init_image (base64)")
            num_frames = int(inp.get("num_frames", 14))
            motion_bucket_id = int(inp.get("motion_bucket_id", 127))
            fps = int(inp.get("fps", 7))
            steps = int(inp.get("steps", 20))
            # Juggernaut 언로드 후 SVD 로드
            unload_main_model()
            log_vram("after unload, before SVD")
            if not load_svd():
                raise RuntimeError("SVD model load failed")
            mp4_b64 = svd_generate(init_image, num_frames, motion_bucket_id, fps, 0.0, steps, seed)
            return {"video": mp4_b64, "status": "success"}
        elif mode == "img2img":
            init_image = inp.get("init_image", "")
            denoise = float(inp.get("denoise", 0.5))
            if not init_image:
                raise ValueError("img2img mode requires init_image (base64)")
            image_tensor = img2img(prompt, negative_prompt, init_image, width, height, steps, cfg_scale, seed, denoise)
        elif mode == "ipadapter":
            face_image = inp.get("face_image", "")
            ipa_strength = float(inp.get("ipa_strength", 0.35))
            if not face_image:
                raise ValueError("ipadapter mode requires face_image (base64)")
            ipa_ok = load_ipadapter()
            if not ipa_ok:
                print("IP-Adapter not available, falling back to txt2img", flush=True)
                image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
            else:
                try:
                    image_tensor = ipadapter_txt2img(
                        prompt, negative_prompt, face_image,
                        width, height, steps, cfg_scale, seed, ipa_strength
                    )
                except Exception as e:
                    import torch
                    print(f"ipadapter txt2img failed ({e}), falling back to txt2img", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        else:
            image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)

        image_b64 = tensor_to_b64(image_tensor)
        try:
            del image_tensor
        except Exception:
            pass
        _force_vram_free()

        return {"image": image_b64, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        _force_vram_free()
        return {"error": str(e), "status": "failed"}


print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
