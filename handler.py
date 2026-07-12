import runpod
import base64
import random
import sys
import os
import gc
import tracemalloc

print("handler.py starting... V73", flush=True)

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

tracemalloc.start()

# ComfyUI --lowvram 강제 (sys.argv 방식 + 나중에 vram_state 직접 설정으로 이중 보장)
for flag in ['--lowvram', '--disable-cuda-malloc', '--disable-smart-memory']:
    if flag not in sys.argv:
        sys.argv.append(flag)

sys.path.insert(0, '/comfyui')

# pinned memory 직접 패치 (--disable-smart-memory 효과 없을 경우 대비)
def _patch_pinned_memory():
    try:
        import comfy.model_management as mm
        # PIN_SHARED_MEMORY 변수 비활성화
        if hasattr(mm, 'PIN_SHARED_MEMORY'):
            old = mm.PIN_SHARED_MEMORY
            mm.PIN_SHARED_MEMORY = False
            print(f"[PATCH] PIN_SHARED_MEMORY: {old} → False", flush=True)
        else:
            print("[PATCH] PIN_SHARED_MEMORY attribute not found", flush=True)
        # pin_memory 함수 no-op으로 대체
        if hasattr(mm, 'pin_memory'):
            mm.pin_memory = lambda tensor: tensor
            print("[PATCH] pin_memory → no-op", flush=True)
        else:
            print("[PATCH] pin_memory function not found", flush=True)
        # should_use_fp16 등 메모리 관련 내부 상태 출력
        for attr in ['current_loaded_models', 'loaded_models_ram', 'vram_state', 'total_vram']:
            if hasattr(mm, attr):
                print(f"[PATCH] mm.{attr} = {getattr(mm, attr)}", flush=True)
    except Exception as e:
        print(f"[PATCH] patch failed: {e}", flush=True)

_patch_pinned_memory()

MODEL_PATH = "/comfyui/models/checkpoints/chilloutmix.safetensors"
CN_PATH = "/comfyui/models/controlnet/control_v11p_sd15_openpose.safetensors"
CN_PATH_FALLBACK = "/comfyui/models/controlnet/control_v11p_sd15_openpose.pth"
IPA_PATH = "/comfyui/models/ipadapter/ip-adapter-plus-face_sd15.safetensors"
IPA_PATH_FALLBACK = "/comfyui/models/ipadapter/ip-adapter-plus-face_sd15.bin"
CLIP_PATH = "/comfyui/models/clip_vision/clip-vit-h-14.safetensors"  # ViT-H: ip-adapter-plus-face_sd15와 매칭

loaded_ipadapter = None
loaded_clip_vision = None

loaded_model = None
loaded_clip = None
loaded_vae = None
loaded_controlnet = None


def _force_vram_free():
    """모든 GPU/RAM 메모리 강제 해제 — 가능한 모든 방법 동원"""
    import gc, torch

    try:
        import comfy.model_management as mm

        # 1. ComfyUI 공식 전체 언로드 함수
        try:
            mm.unload_all_models()
            print(f"[VRAM] unload_all_models OK", flush=True)
        except Exception as e:
            print(f"[VRAM] unload_all_models error: {e}", flush=True)

        # 2. current_loaded_models 수동 언로드 + clear
        loaded_count = len(mm.current_loaded_models)
        print(f"[VRAM] current_loaded_models count: {loaded_count}", flush=True)
        for lm in list(mm.current_loaded_models):
            try:
                lm.model_unload()
            except Exception as e:
                print(f"[VRAM] model_unload error: {e}", flush=True)
        mm.current_loaded_models.clear()
        print(f"[VRAM] current_loaded_models cleared", flush=True)

        # 3. VRAM 전체 강제 해제
        try:
            mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
            print(f"[VRAM] free_memory OK", flush=True)
        except Exception as e:
            print(f"[VRAM] free_memory error: {e}", flush=True)

        # 4. soft_empty_cache
        try:
            mm.soft_empty_cache()
            print(f"[VRAM] soft_empty_cache OK", flush=True)
        except Exception as e:
            print(f"[VRAM] soft_empty_cache error: {e}", flush=True)

    except Exception as e:
        print(f"[VRAM] mm cleanup error: {e}", flush=True)

    # 5. 글로벌 참조 해제 → Python GC가 RAM 수거
    global loaded_model, loaded_clip, loaded_vae, loaded_ipadapter, loaded_clip_vision, loaded_controlnet
    for gname, gobj in [
        ('loaded_model', loaded_model), ('loaded_clip', loaded_clip),
        ('loaded_vae', loaded_vae), ('loaded_ipadapter', loaded_ipadapter),
        ('loaded_clip_vision', loaded_clip_vision),
    ]:
        if gobj is None:
            continue
        try:
            inner = getattr(gobj, 'model', None)
            if inner is not None and hasattr(inner, 'to'):
                inner.to('cpu')
            elif hasattr(gobj, 'to'):
                gobj.to('cpu')
        except Exception as e:
            print(f"[VRAM] {gname} cpu move error: {e}", flush=True)

    loaded_model = None
    loaded_clip = None
    loaded_vae = None
    loaded_ipadapter = None
    loaded_clip_vision = None
    loaded_controlnet = None

    # 6. GC + VRAM 캐시 완전 초기화
    torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    log_vram("after _force_vram_free")


def load_model():
    global loaded_model, loaded_clip, loaded_vae
    if loaded_model is not None:
        return
    print("Loading ComfyUI modules...", flush=True)
    from nodes import CheckpointLoaderSimple

    # vram_state 직접 LOW_VRAM으로 설정 (sys.argv 방식 이중 보장)
    try:
        import comfy.model_management as mm
        print(f"[V64] vram_state before override: {mm.vram_state}", flush=True)
        mm.vram_state = mm.VRAMState.LOW_VRAM
        print(f"[V64] vram_state after override: {mm.vram_state}", flush=True)
    except Exception as e:
        print(f"[V64] vram_state override failed: {e}", flush=True)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model_size_gb = os.path.getsize(MODEL_PATH) / (1024 ** 3)
    print(f"[V64] Model file size: {model_size_gb:.2f} GB ({MODEL_PATH})", flush=True)

    print(f"Loading model from {MODEL_PATH}...", flush=True)
    loader = CheckpointLoaderSimple()
    loaded_model, loaded_clip, loaded_vae = loader.load_checkpoint("chilloutmix.safetensors")
    print("Model loaded!", flush=True)
    log_vram("after load_model")
    # CLIP Vision 미리 로드 — VRAM 상주로 OOM spike 방지
    load_ipadapter()


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
    ipa_filename = None
    if os.path.exists(IPA_PATH):
        ipa_filename = "ip-adapter-plus-face_sd15.safetensors"
    elif os.path.exists(IPA_PATH_FALLBACK):
        ipa_filename = "ip-adapter-plus-face_sd15.bin"
    if not ipa_filename or not os.path.exists(CLIP_PATH):
        print("IP-Adapter or CLIP Vision model not found, skipping.", flush=True)
        return False
    try:
        from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterModelLoader
        from nodes import CLIPVisionLoader
        print(f"Loading IP-Adapter model ({ipa_filename})...", flush=True)
        loader = IPAdapterModelLoader()
        loaded_ipadapter = loader.load_ipadapter_model(ipa_filename)[0]
        clip_loader = CLIPVisionLoader()
        loaded_clip_vision = clip_loader.load_clip("clip-vit-h-14.safetensors")[0]
        print("IP-Adapter loaded!", flush=True)
        return True
    except Exception as e:
        print(f"IP-Adapter load failed: {e}", flush=True)
        return False


def ipadapter_img2img(prompt, negative_prompt, pose_image_b64, face_image_b64, width, height, steps, cfg_scale, seed, ipa_strength=0.7, denoise=0.85):
    import torch
    import numpy as np
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    from PIL import Image
    from io import BytesIO
    from nodes import CLIPTextEncode, KSampler, VAEDecode, VAEEncode
    # 최신 IPAdapterPlus에서 사용 가능한 클래스 동적 탐색
    ipa_module = __import__(
        'custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus',
        fromlist=['IPAdapterAdvanced', 'IPAdapterPlus', 'IPAdapter']
    )
    # 우선순위: IPAdapterAdvanced > IPAdapterPlus > IPAdapter
    IPAdapterClass = (
        getattr(ipa_module, 'IPAdapterAdvanced', None) or
        getattr(ipa_module, 'IPAdapterPlus', None) or
        getattr(ipa_module, 'IPAdapter', None)
    )
    if IPAdapterClass is None:
        raise ImportError("No usable IPAdapter class found in IPAdapterPlus module")
    print(f"Using IPAdapter class: {IPAdapterClass.__name__}", flush=True)

    def b64_to_tensor(b64, label="image", size=None):
        import urllib.request as _ur
        target = size or (width, height)
        if b64.startswith('http://') or b64.startswith('https://'):
            print(f"[b64_to_tensor] downloading {label} from URL: {b64[:80]}", flush=True)
            try:
                req = _ur.Request(b64, headers={'User-Agent': 'Mozilla/5.0'})
                with _ur.urlopen(req, timeout=15) as r:
                    raw = r.read()
                print(f"[b64_to_tensor] downloaded {label}: {len(raw)} bytes, status={r.status}", flush=True)
                img = Image.open(BytesIO(raw)).convert("RGB").resize(target, Image.LANCZOS)
            except Exception as e:
                print(f"[b64_to_tensor] FAILED to load {label} from URL: {e}", flush=True)
                raise
        else:
            print(f"[b64_to_tensor] decoding {label} from base64 (len={len(b64)})", flush=True)
            try:
                if ',' in b64:
                    b64 = b64.split(',', 1)[1]
                b64 = b64.strip()
                b64 = b64.encode('ascii', errors='ignore').decode('ascii')
                b64 += '=' * (-len(b64) % 4)
                img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB").resize(target, Image.LANCZOS)
            except Exception as e:
                print(f"[b64_to_tensor] FAILED to decode {label} base64: {e}", flush=True)
                raise
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    pose_tensor = b64_to_tensor(pose_image_b64, label="pose_image")
    face_tensor = b64_to_tensor(face_image_b64, label="face_image", size=(224, 224))

    # pose 이미지를 init_image로 인코딩
    vae_encoder = VAEEncode()
    try:
        with torch.inference_mode(False):
            latent = vae_encoder.encode(loaded_vae, pose_tensor.clone())[0]
    except Exception as e:
        print(f"VAE encode failed ({e}), falling back to img2img", flush=True)
        return img2img(prompt, negative_prompt, pose_image_b64, denoise, width, height, steps, cfg_scale, seed)

    # 텍스트 인코딩
    clip_encoder = CLIPTextEncode()
    positive = clip_encoder.encode(loaded_clip, prompt)[0]
    negative_cond = clip_encoder.encode(loaded_clip, negative_prompt)[0]

    # IP-Adapter 적용 (얼굴 이미지 기반)
    ipa_node = IPAdapterClass()
    result = ipa_node.apply_ipadapter(
        model=loaded_model, ipadapter=loaded_ipadapter,
        clip_vision=loaded_clip_vision, image=face_tensor,
        weight=ipa_strength, weight_type="original",
        start_at=0.0, end_at=1.0
    )
    model_with_ipa = result[0]

    try:
        sampler = KSampler()
        sampled = sampler.sample(
            model_with_ipa, seed, steps, cfg_scale,
            "euler_ancestral", "karras",
            positive, negative_cond, latent, denoise=denoise
        )[0]

        decoder = VAEDecode()
        result_img = decoder.decode(loaded_vae, sampled)[0]
        return result_img
    finally:
        for _var in ['model_with_ipa', 'positive', 'negative_cond', 'latent',
                     'sampled', 'pose_tensor', 'face_tensor', 'ipa_node']:
            try:
                if _var in dir():
                    del _var
            except Exception:
                pass
        gc.collect()
        _force_vram_free()


def ipadapter_txt2img(prompt, negative_prompt, face_image_b64, width, height, steps, cfg_scale, seed, ipa_strength=0.35):
    """txt2img + IP-Adapter 얼굴 conditioning (pose_image 없이)"""
    import torch
    import numpy as np
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # VRAM 여유 확인 — 2GB 미만이면 IPA 스킵
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024 / 1024 / 1024
    print(f"VRAM before IPA: {int(free/1024/1024)}MB free / {int(total/1024/1024)}MB total", flush=True)
    if free_gb < 2.0:
        print(f"VRAM too low ({free_gb:.1f}GB free), skipping IPA → txt2img", flush=True)
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
        raise ImportError("No usable IPAdapter class found in IPAdapterPlus module")
    print(f"IPA txt2img using: {IPAdapterClass.__name__}", flush=True)

    print(f"[ipadapter_txt2img] decoding face_image base64 (len={len(face_image_b64)})", flush=True)
    if ',' in face_image_b64:
        face_image_b64 = face_image_b64.split(',', 1)[1]
    face_b = face_image_b64.strip()
    face_b = face_b.encode('ascii', errors='ignore').decode('ascii')
    face_b += '=' * (-len(face_b) % 4)
    face_img = Image.open(BytesIO(base64.b64decode(face_b))).convert("RGB").resize((224, 224), Image.LANCZOS)
    face_arr = np.array(face_img).astype(np.float32) / 255.0
    face_tensor = torch.from_numpy(face_arr).unsqueeze(0)
    print(f"[ipadapter_txt2img] face_image decoded OK: {face_img.size}", flush=True)

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
        for _var in ['model_with_ipa', 'positive', 'negative_cond', 'latent',
                     'sampled', 'face_tensor', 'ipa_node']:
            try:
                if _var in dir():
                    del _var
            except Exception:
                pass
        gc.collect()
        _force_vram_free()


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


def img2img(prompt, negative_prompt, init_image_b64, denoising_strength, width, height, steps, cfg_scale, seed):
    import torch
    import numpy as np
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    from PIL import Image
    from io import BytesIO
    from nodes import CLIPTextEncode, KSampler, VAEDecode, VAEEncode

    import urllib.request as _ur
    if init_image_b64.startswith('http://') or init_image_b64.startswith('https://'):
        with _ur.urlopen(init_image_b64, timeout=15) as r:
            pil_img = Image.open(BytesIO(r.read())).convert("RGB")
    else:
        if ',' in init_image_b64:
            init_image_b64 = init_image_b64.split(',', 1)[1]
        init_image_b64 = init_image_b64.strip()
        init_image_b64 = init_image_b64.encode('ascii', errors='ignore').decode('ascii')
        init_image_b64 += '=' * (-len(init_image_b64) % 4)  # 패딩 보정
        img_bytes = base64.b64decode(init_image_b64)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    pil_img = pil_img.resize((width, height), Image.LANCZOS)

    np_img = np.array(pil_img).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np_img).unsqueeze(0)

    vae_encoder = VAEEncode()
    with torch.inference_mode(False):
        latent = vae_encoder.encode(loaded_vae, image_tensor.clone())[0]

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


def log_vram(label=""):
    try:
        import torch, psutil, os
        free, total = torch.cuda.mem_get_info()
        proc = psutil.Process(os.getpid())
        ram_gb = proc.memory_info().rss / 1024**3
        print(f"[VRAM]{' ' + label if label else ''}: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total | RAM: {ram_gb:.2f}GB", flush=True)
    except Exception:
        pass


def log_mem_detail(label=""):
    """상세 메모리 진단 — tensor 수/크기, tracemalloc top5, mm 상태"""
    try:
        import torch
        # 1. tensor 개수 + 총 크기
        tensors = [o for o in gc.get_objects() if isinstance(o, torch.Tensor)]
        cpu_t = [t for t in tensors if not t.is_cuda]
        gpu_t = [t for t in tensors if t.is_cuda]
        cpu_gb = sum(t.element_size() * t.nelement() for t in cpu_t) / 1024**3
        gpu_gb = sum(t.element_size() * t.nelement() for t in gpu_t) / 1024**3
        print(f"[MEM-{label}] Tensors CPU: {len(cpu_t)}개 {cpu_gb:.2f}GB | GPU: {len(gpu_t)}개 {gpu_gb:.2f}GB", flush=True)
    except Exception as e:
        print(f"[MEM-{label}] tensor scan error: {e}", flush=True)

    try:
        # 2. tracemalloc top5 — 어느 라인에서 RAM 많이 쓰는지
        snapshot = tracemalloc.take_snapshot()
        for i, stat in enumerate(snapshot.statistics('lineno')[:5]):
            print(f"[MEM-{label}] top{i+1}: {stat}", flush=True)
    except Exception as e:
        print(f"[MEM-{label}] tracemalloc error: {e}", flush=True)

    try:
        # 3. ComfyUI current_loaded_models 상태
        import comfy.model_management as mm
        loaded = mm.current_loaded_models
        print(f"[MEM-{label}] mm.current_loaded_models: {len(loaded)}개", flush=True)
        for i, lm in enumerate(loaded):
            model_name = getattr(getattr(lm, 'model', None), '__class__', type(lm)).__name__
            print(f"[MEM-{label}]   [{i}] {model_name}", flush=True)
    except Exception as e:
        print(f"[MEM-{label}] mm state error: {e}", flush=True)


def handler(job):
    try:
        inp = job["input"]
        mode = inp.get("mode", "txt2img")  # "txt2img" | "img2img" | "controlnet"
        prompt = inp.get("prompt", "")
        negative_prompt = inp.get("negative_prompt", "")
        width = inp.get("width", 512)
        height = inp.get("height", 768)
        steps = inp.get("steps", 20)
        cfg_scale = inp.get("cfg_scale", 7)
        seed = inp.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        load_model()
        log_vram("before generation")
        log_mem_detail("before")

        print(f"Mode={mode}, {width}x{height}, steps={steps}, seed={seed}", flush=True)

        if mode == "ipadapter":
            pose_image = inp.get("pose_image", "")
            face_image = inp.get("face_image", "")
            ipa_strength = float(inp.get("ipa_strength", 0.35))
            denoise = float(inp.get("denoising_strength", 0.85))
            if not face_image:
                raise ValueError("ipadapter mode requires face_image (base64)")
            ipa_ok = load_ipadapter()
            if not ipa_ok:
                print("IP-Adapter not available, falling back to txt2img", flush=True)
                image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
            elif pose_image:
                # pose_image 있으면 img2img + IP-Adapter
                try:
                    image_tensor = ipadapter_img2img(
                        prompt, negative_prompt, pose_image, face_image,
                        width, height, steps, cfg_scale, seed, ipa_strength, denoise
                    )
                except Exception as e:
                    import torch, gc
                    print(f"ipadapter img2img failed ({e}), falling back to ipadapter txt2img", flush=True)
                    gc.collect(); torch.cuda.empty_cache()
                    try:
                        image_tensor = ipadapter_txt2img(
                            prompt, negative_prompt, face_image,
                            width, height, steps, cfg_scale, seed, ipa_strength
                        )
                    except Exception as e2:
                        print(f"ipadapter txt2img also failed ({e2}), falling back to txt2img", flush=True)
                        gc.collect(); torch.cuda.empty_cache()
                        image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
            else:
                # pose_image 없으면 txt2img + IP-Adapter (얼굴만 conditioning)
                try:
                    image_tensor = ipadapter_txt2img(
                        prompt, negative_prompt, face_image,
                        width, height, steps, cfg_scale, seed, ipa_strength
                    )
                except Exception as e:
                    import torch, gc
                    print(f"ipadapter txt2img failed ({e}), clearing VRAM and falling back to txt2img", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    image_tensor = txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
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
        try:
            del image_tensor
        except Exception:
            pass
        _force_vram_free()
        log_mem_detail("after")
        return {"image": image_b64, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        _force_vram_free()
        return {"error": str(e), "status": "failed"}


print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
