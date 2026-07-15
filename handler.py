import runpod
import base64
import random
import sys
import os
import gc
import tracemalloc

print("handler.py starting... V95", flush=True)

# V89: IPA job 카운터 — 70개마다 worker 재시작 (503GB RAM / 5.8GB per job = ~86, 여유 16개)
_ipa_job_count = 0

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

tracemalloc.start()

# ComfyUI --lowvram 강제 (sys.argv 방식 + 나중에 vram_state 직접 설정으로 이중 보장)
# V88: --disable-async-offload 추가 → async weight offloading이 float32 cast buffer 누수의 근본 원인
for flag in ['--lowvram', '--disable-cuda-malloc', '--disable-smart-memory', '--disable-async-offload']:
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
        # V88: async weight offloading 직접 비활성화
        # ENABLE_ASYNC_OFFLOAD 또는 동등한 변수
        for async_attr in ['ENABLE_ASYNC_OFFLOAD', 'async_offload_enabled', 'use_async_offload']:
            if hasattr(mm, async_attr):
                old_v = getattr(mm, async_attr)
                setattr(mm, async_attr, False)
                print(f"[V88-PATCH] {async_attr}: {old_v} → False", flush=True)
        # V88: stream 수 0으로 (async offloading용 CUDA stream 제거)
        for stream_attr in ['OFFLOAD_WORKER_STREAMS', 'offload_streams', 'stream_count']:
            if hasattr(mm, stream_attr):
                val = getattr(mm, stream_attr)
                print(f"[V88-PATCH] {stream_attr} found: {type(val).__name__} len={len(val) if hasattr(val, '__len__') else 'N/A'}", flush=True)
        # V88: get_torch_device_stream 패치로 stream None 반환 → async offloading 중단
        if hasattr(mm, 'get_torch_device_stream'):
            original_stream_fn = mm.get_torch_device_stream
            def _no_stream(*args, **kwargs):
                return None
            mm.get_torch_device_stream = _no_stream
            print("[V88-PATCH] get_torch_device_stream → None (async offload 차단)", flush=True)
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
    """VRAM 캐시만 정리 — ComfyUI 모델 추적 목록은 건드리지 않음"""
    import gc, torch

    # V75: unload_all_models/clear 제거
    # LOW_VRAM 모드에서 ComfyUI가 자동으로 모델을 CPU로 오프로드함.
    # unload_all_models() + current_loaded_models.clear() 를 호출하면
    # 다음 job에서 ComfyUI가 새 LoadedModel 래퍼를 생성 → 추적 tensor 누적.
    try:
        import comfy.model_management as mm
        loaded = len(mm.current_loaded_models)
        print(f"[VRAM] current_loaded_models: {loaded}개 (유지)", flush=True)
        try:
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[VRAM] soft_empty_cache error: {e}", flush=True)
    except Exception as e:
        print(f"[VRAM] mm state error: {e}", flush=True)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # V86: async weight offloading cast buffer 클리어 (STREAM_CAST_BUFFERS 누수 방지)
    try:
        import comfy.model_management as mm
        if hasattr(mm, 'reset_cast_buffers'):
            mm.reset_cast_buffers()
            print("[V86] reset_cast_buffers OK", flush=True)
    except Exception as e:
        print(f"[V86] reset_cast_buffers error: {e}", flush=True)

    # V87-A: CLIPVision을 current_loaded_models에서 제거 → ComfyUI offload 타깃 제외
    # 목적: CLIPVision이 VRAM→CPU cast buffer로 offload되는 것 자체를 막음
    # (reset_cast_buffers가 ModelPatcher 참조 때문에 해제 못하는 근본 원인 차단)
    try:
        import comfy.model_management as mm
        if loaded_clip_vision is not None:
            before = len(mm.current_loaded_models)
            # CLIPVision 모델이 current_loaded_models에 있으면 제거
            mm.current_loaded_models[:] = [
                lm for lm in mm.current_loaded_models
                if getattr(lm, 'model', None) is not loaded_clip_vision
                and getattr(getattr(lm, 'model', None), 'load_model', None) is not loaded_clip_vision
                and lm is not loaded_clip_vision
            ]
            after = len(mm.current_loaded_models)
            print(f"[V87-A] current_loaded_models: {before}→{after} (CLIPVision 제거)", flush=True)
    except Exception as e:
        print(f"[V87-A] error: {e}", flush=True)

    # V87-B: STREAM_CAST_BUFFERS 직접 접근하여 CLIPVision 관련 텐서만 강제 해제
    try:
        import comfy.model_management as mm
        import torch
        if hasattr(mm, 'STREAM_CAST_BUFFERS') and mm.STREAM_CAST_BUFFERS:
            before_count = len(mm.STREAM_CAST_BUFFERS)
            before_gb = sum(t.element_size() * t.nelement() for t in mm.STREAM_CAST_BUFFERS) / 1024**3
            # 전체 클리어 (reset_cast_buffers와 동일하지만 del로 실제 참조 제거)
            for t in list(mm.STREAM_CAST_BUFFERS):
                del t
            mm.STREAM_CAST_BUFFERS.clear()
            gc.collect()
            print(f"[V87-B] STREAM_CAST_BUFFERS cleared: {before_count}개 {before_gb:.2f}GB", flush=True)
        else:
            scb_count = len(mm.STREAM_CAST_BUFFERS) if hasattr(mm, 'STREAM_CAST_BUFFERS') else -1
            print(f"[V87-B] STREAM_CAST_BUFFERS count={scb_count}", flush=True)
    except Exception as e:
        print(f"[V87-B] error: {e}", flush=True)

    # V87-C: gc 2차 collect + CPU tensor 총량 리포트
    try:
        import torch
        gc.collect()
        gc.collect()
        all_cpu = [t for t in gc.get_objects() if isinstance(t, torch.Tensor) and not t.is_cuda]
        total_gb = sum(t.element_size() * t.nelement() for t in all_cpu) / 1024**3
        print(f"[V87-C] CPU tensors after full cleanup: {len(all_cpu)}개 {total_gb:.2f}GB", flush=True)
    except Exception as e:
        print(f"[V87-C] error: {e}", flush=True)

    # V88: LoadedModel 내부 weight cache 강제 클리어
    # async offloading이 float32 cast 텐서를 LoadedModel 내부 list에 보관하는 것 차단
    try:
        import comfy.model_management as mm
        cleared_any = False
        for lm in mm.current_loaded_models:
            # weights_loaded: async offloading이 loaded weight 참조를 추적하는 list
            for attr in ['weights_loaded', 'model_weights_loaded', '_weights', 'offload_weights',
                         'loaded_weights', 'cast_weights', 'stream_buffer', 'cpu_weights']:
                obj = getattr(lm, attr, None)
                if obj is not None and isinstance(obj, list) and len(obj) > 0:
                    before_len = len(obj)
                    obj.clear()
                    print(f"[V88] lm.{attr} cleared: {before_len}개", flush=True)
                    cleared_any = True
                elif obj is not None and isinstance(obj, dict) and len(obj) > 0:
                    before_len = len(obj)
                    obj.clear()
                    print(f"[V88] lm.{attr}(dict) cleared: {before_len}개", flush=True)
                    cleared_any = True
            # model 내부도 확인
            m = getattr(lm, 'model', None)
            if m is not None:
                for attr in ['weights_loaded', 'offload_weights', 'cast_weights']:
                    obj2 = getattr(m, attr, None)
                    if obj2 is not None and isinstance(obj2, list) and len(obj2) > 0:
                        obj2.clear()
                        print(f"[V88] lm.model.{attr} cleared", flush=True)
                        cleared_any = True
        if not cleared_any:
            print(f"[V88] no weight cache found in {len(mm.current_loaded_models)} LoadedModels", flush=True)
        gc.collect()
        gc.collect()
        all_cpu2 = [t for t in gc.get_objects() if isinstance(t, torch.Tensor) and not t.is_cuda]
        total_gb2 = sum(t.element_size() * t.nelement() for t in all_cpu2) / 1024**3
        print(f"[V88] CPU tensors after LoadedModel clear: {len(all_cpu2)}개 {total_gb2:.2f}GB", flush=True)
    except Exception as e:
        print(f"[V88] error: {e}", flush=True)

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
    # V85: job 시작 전 CPU tensor id 스냅샷
    try:
        _before_tensor_ids = set(id(t) for t in gc.get_objects() if isinstance(t, torch.Tensor) and not t.is_cuda)
    except Exception:
        _before_tensor_ids = set()
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
    del result

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
        # V81: model_with_ipa 내부 참조 완전 클리어
        try:
            import comfy.model_management as mm
            before = len(mm.current_loaded_models)
            mm.current_loaded_models[:] = [
                lm for lm in mm.current_loaded_models
                if getattr(lm, 'model', None) is not model_with_ipa
                and lm is not model_with_ipa
            ]
            after = len(mm.current_loaded_models)
            if before != after:
                print(f"[V76] removed model_with_ipa: {before}→{after}", flush=True)
        except Exception as e:
            print(f"[V76] cleanup error: {e}", flush=True)

        try:
            if hasattr(model_with_ipa, 'model_options') and isinstance(model_with_ipa.model_options, dict):
                to = model_with_ipa.model_options.get('transformer_options', {})
                to.clear()
                model_with_ipa.model_options.clear()
            if hasattr(model_with_ipa, 'patches') and isinstance(model_with_ipa.patches, dict):
                model_with_ipa.patches.clear()
            if hasattr(model_with_ipa, 'object_patches') and isinstance(model_with_ipa.object_patches, dict):
                model_with_ipa.object_patches.clear()
            if hasattr(model_with_ipa, 'parent'):
                model_with_ipa.parent = None
            print("[V81] model_with_ipa cleared", flush=True)
        except Exception as e:
            print(f"[V81] clear error: {e}", flush=True)

        # V82: 원본 loaded_model에 누적된 patches/transformer_options 클리어
        try:
            lm_p = getattr(loaded_model, 'patches', None)
            lm_to = (getattr(loaded_model, 'model_options', {}) or {}).get('transformer_options', {})
            n_p = len(lm_p) if lm_p else 0
            n_to = len(lm_to) if lm_to else 0
            print(f"[V82] loaded_model patches={n_p} transformer_options={n_to}", flush=True)
            if n_p > 0:
                lm_p.clear()
            if n_to > 0:
                lm_to.clear()
        except Exception as e:
            print(f"[V82] loaded_model cleanup error: {e}", flush=True)

        # V83: loaded_clip_vision patches/transformer_options 조사 + 클리어
        try:
            cv_p = getattr(loaded_clip_vision, 'patches', None)
            cv_to = (getattr(loaded_clip_vision, 'model_options', {}) or {}).get('transformer_options', {})
            n_cvp = len(cv_p) if cv_p else 0
            n_cvto = len(cv_to) if cv_to else 0
            print(f"[V83] loaded_clip_vision patches={n_cvp} transformer_options={n_cvto}", flush=True)
            if n_cvp > 0:
                cv_p.clear()
            if n_cvto > 0:
                cv_to.clear()
        except Exception as e:
            print(f"[V83] clip_vision cleanup error: {e}", flush=True)

        # V83: loaded_ipadapter 내부 구조 로깅
        try:
            if isinstance(loaded_ipadapter, dict):
                ipa_keys = list(loaded_ipadapter.keys())
                print(f"[V83] loaded_ipadapter keys={ipa_keys}", flush=True)
            else:
                ipa_p = getattr(loaded_ipadapter, 'patches', None)
                n_ipap = len(ipa_p) if ipa_p else 0
                print(f"[V83] loaded_ipadapter type={type(loaded_ipadapter).__name__} patches={n_ipap}", flush=True)
        except Exception as e:
            print(f"[V83] ipadapter inspect error: {e}", flush=True)

        del model_with_ipa, positive, negative_cond, latent, ipa_node
        try:
            del sampled
        except Exception:
            pass
        del pose_tensor, face_tensor
        gc.collect()

        # V85+V90: 누수 tensor → list → cell → 클로저 함수 역추적
        try:
            import comfy.model_management as _mm_diag
            import types
            all_cpu = [t for t in gc.get_objects() if isinstance(t, torch.Tensor) and not t.is_cuda]
            new_tensors = [t for t in all_cpu if id(t) not in _before_tensor_ids]
            print(f"[V85] new CPU tensors vs job start: {len(new_tensors)}개", flush=True)
            if new_tensors:
                new_tensors_sorted = sorted(new_tensors, key=lambda t: t.nelement() * t.element_size(), reverse=True)
                sample = new_tensors_sorted[0]
                mb = sample.element_size() * sample.nelement() / 1024**2
                print(f"[V85] #1 shape={list(sample.shape)} dtype={sample.dtype} {mb:.1f}MB", flush=True)
                refs = gc.get_referrers(sample)
                for r in refs:
                    if isinstance(r, list):
                        list_owners = gc.get_referrers(r)
                        for lo in list_owners[:6]:
                            if isinstance(lo, types.CellType):
                                # cell 소유자 함수 역추적
                                cell_owners = gc.get_referrers(lo)
                                for co in cell_owners[:4]:
                                    if callable(co) and hasattr(co, '__name__'):
                                        mod = getattr(co, '__module__', '?')
                                        print(f"[V90] tensor→list→cell→func: {mod}.{co.__name__}", flush=True)
                                    elif isinstance(co, tuple):
                                        # __closure__ tuple — 소유 함수 찾기
                                        closure_owners = gc.get_referrers(co)
                                        for fo in closure_owners[:3]:
                                            if callable(fo) and hasattr(fo, '__name__'):
                                                mod = getattr(fo, '__module__', '?')
                                                print(f"[V90] tensor→list→cell→closure→func: {mod}.{fo.__name__}", flush=True)
                                            elif hasattr(fo, '__class__'):
                                                print(f"[V90] tensor→list→cell→closure→{type(fo).__name__}", flush=True)
                            elif isinstance(lo, dict):
                                mm_vars = vars(_mm_diag)
                                if lo is mm_vars:
                                    matches = [k for k, v in mm_vars.items() if v is r]
                                    print(f"[V90] tensor→list→mm.{matches}", flush=True)
                                else:
                                    dict_owners = gc.get_referrers(lo)
                                    for do in dict_owners[:2]:
                                        if hasattr(do, '__dict__') and do.__dict__ is lo:
                                            attrs = [k for k, v in lo.items() if v is r]
                                            print(f"[V90] tensor→list→{type(do).__name__}.{attrs}", flush=True)
                                            break
                                    else:
                                        print(f"[V90] tensor→list→dict(keys={list(lo.keys())[:5]})", flush=True)
                            elif isinstance(lo, type(_mm_diag)):
                                print(f"[V90] tensor→list→module({lo.__name__})", flush=True)
                            else:
                                print(f"[V90] tensor→list→{type(lo).__name__}", flush=True)
                    elif isinstance(r, dict):
                        mm_vars = vars(_mm_diag)
                        if r is mm_vars:
                            matches = [k for k, v in mm_vars.items() if v is sample]
                            print(f"[V90] tensor→directly in mm.{matches}", flush=True)
                        else:
                            print(f"[V90] tensor→dict(keys={list(r.keys())[:5]})", flush=True)
                    elif isinstance(r, tuple):
                        tuple_owners = gc.get_referrers(r)
                        for to in tuple_owners[:2]:
                            print(f"[V90] tensor→tuple→{type(to).__name__}", flush=True)
        except Exception as e:
            print(f"[V90] referrer error: {e}", flush=True)

        _force_vram_free()


def ipadapter_txt2img(prompt, negative_prompt, face_image_b64, width, height, steps, cfg_scale, seed, ipa_strength=0.35):
    """txt2img + IP-Adapter 얼굴 conditioning (pose_image 없이)"""
    import torch
    import numpy as np
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # V85: job 시작 전 CPU tensor id 스냅샷
    try:
        _before_tensor_ids = set(id(t) for t in gc.get_objects() if isinstance(t, torch.Tensor) and not t.is_cuda)
    except Exception:
        _before_tensor_ids = set()

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
        # V81: model_with_ipa 내부 참조 완전 클리어 (closure/embedding 해제)
        try:
            import comfy.model_management as mm
            before = len(mm.current_loaded_models)
            mm.current_loaded_models[:] = [
                lm for lm in mm.current_loaded_models
                if getattr(lm, 'model', None) is not model_with_ipa
                and lm is not model_with_ipa
            ]
            after = len(mm.current_loaded_models)
            if before != after:
                print(f"[V76] removed model_with_ipa: {before}→{after}", flush=True)
        except Exception as e:
            print(f"[V76] cleanup error: {e}", flush=True)

        try:
            # V84: transformer_options 내용만 클리어, 키는 유지 (model_options.clear()하면 __del__이 KeyError로 죽어서 tensor 해제 안 됨)
            if hasattr(model_with_ipa, 'model_options') and isinstance(model_with_ipa.model_options, dict):
                to = model_with_ipa.model_options.get('transformer_options', {})
                to.clear()
                # model_with_ipa.model_options.clear() ← 제거: transformer_options 키 자체를 지우면 __del__ KeyError
            if hasattr(model_with_ipa, 'patches') and isinstance(model_with_ipa.patches, dict):
                model_with_ipa.patches.clear()
            if hasattr(model_with_ipa, 'object_patches') and isinstance(model_with_ipa.object_patches, dict):
                model_with_ipa.object_patches.clear()
            if hasattr(model_with_ipa, 'parent'):
                model_with_ipa.parent = None
            print("[V84] model_with_ipa cleared", flush=True)
        except Exception as e:
            print(f"[V84] clear error: {e}", flush=True)

        # V82: 원본 loaded_model에 누적된 patches/transformer_options 클리어
        try:
            lm_p = getattr(loaded_model, 'patches', None)
            lm_to = (getattr(loaded_model, 'model_options', {}) or {}).get('transformer_options', {})
            n_p = len(lm_p) if lm_p else 0
            n_to = len(lm_to) if lm_to else 0
            print(f"[V82] loaded_model patches={n_p} transformer_options={n_to}", flush=True)
            if n_p > 0:
                lm_p.clear()
            if n_to > 0:
                lm_to.clear()
        except Exception as e:
            print(f"[V82] loaded_model cleanup error: {e}", flush=True)

        # V83: loaded_clip_vision patches/transformer_options 조사 + 클리어
        try:
            cv_p = getattr(loaded_clip_vision, 'patches', None)
            cv_to = (getattr(loaded_clip_vision, 'model_options', {}) or {}).get('transformer_options', {})
            n_cvp = len(cv_p) if cv_p else 0
            n_cvto = len(cv_to) if cv_to else 0
            print(f"[V83] loaded_clip_vision patches={n_cvp} transformer_options={n_cvto}", flush=True)
            if n_cvp > 0:
                cv_p.clear()
            if n_cvto > 0:
                cv_to.clear()
        except Exception as e:
            print(f"[V83] clip_vision cleanup error: {e}", flush=True)

        # V83: loaded_ipadapter 내부 구조 로깅
        try:
            if isinstance(loaded_ipadapter, dict):
                ipa_keys = list(loaded_ipadapter.keys())
                print(f"[V83] loaded_ipadapter keys={ipa_keys}", flush=True)
            else:
                ipa_p = getattr(loaded_ipadapter, 'patches', None)
                n_ipap = len(ipa_p) if ipa_p else 0
                print(f"[V83] loaded_ipadapter type={type(loaded_ipadapter).__name__} patches={n_ipap}", flush=True)
        except Exception as e:
            print(f"[V83] ipadapter inspect error: {e}", flush=True)

        del model_with_ipa, positive, negative_cond, latent, ipa_node
        try:
            del sampled
        except Exception:
            pass
        del face_tensor
        gc.collect()

        try:
            cpu_t = [o for o in gc.get_objects() if isinstance(o, torch.Tensor) and not o.is_cuda]
            print(f"[V82] CPU tensors after del+collect: {len(cpu_t)}개 {sum(t.element_size()*t.nelement() for t in cpu_t)/1024**3:.2f}GB", flush=True)
        except Exception:
            pass

        # V97: list len=5 정체 확인 + IPA attn_processors 강제 초기화
        try:
            import sys

            all_cpu = [t for t in gc.get_objects() if isinstance(t, torch.Tensor) and not t.is_cuda]
            new_tensors = [t for t in all_cpu if id(t) not in _before_tensor_ids]
            print(f"[V97] new CPU tensors: {len(new_tensors)}개", flush=True)

            if new_tensors:
                new_tensors_sorted = sorted(new_tensors, key=lambda t: t.nelement() * t.element_size(), reverse=True)
                top5_slice = new_tensors_sorted[:5]  # 명시적으로 저장해서 id 비교용
                sample = new_tensors_sorted[0]
                mb = sample.element_size() * sample.nelement() / 1024**2
                print(f"[V97] #1 shape={list(sample.shape)} dtype={sample.dtype} {mb:.1f}MB", flush=True)

                # list len=5 의 정체 추적
                refs = gc.get_referrers(sample)
                for ref in refs:
                    if isinstance(ref, list) and len(ref) == 5:
                        if ref is top5_slice:
                            print(f"[V97] list len=5 → 우리 진단코드 top5_slice (무시 가능)", flush=True)
                        else:
                            print(f"[V97] list len=5 → 외부 리스트 발견! id={id(ref)}", flush=True)
                            # 이 리스트를 누가 붙잡고 있는지
                            list_owners = gc.get_referrers(ref)
                            for lo in list_owners[:5]:
                                lo_type = type(lo).__name__
                                lo_repr = ""
                                if isinstance(lo, dict):
                                    keys = [k for k in lo.keys() if not k.startswith('__')][:5]
                                    lo_repr = f"keys={keys}"
                                    for owner in gc.get_referrers(lo)[:3]:
                                        if hasattr(owner, '__dict__') and owner.__dict__ is lo:
                                            lo_repr += f" owner={type(owner).__name__}({getattr(owner, '__module__', '?')})"
                                            break
                                elif isinstance(lo, list):
                                    lo_repr = f"len={len(lo)}"
                                elif hasattr(lo, '__class__'):
                                    lo_repr = f"class={type(lo).__name__} mod={getattr(lo, '__module__', '?')}"
                                print(f"[V97]   list_owner: {lo_type} {lo_repr}", flush=True)

                # IPA attn_processors 확인
                try:
                    unet = loaded_model.model.diffusion_model
                    procs = unet.attn_processors
                    ipa_count = sum(1 for v in procs.values() if 'IP' in type(v).__name__ or 'ip' in type(v).__name__.lower())
                    print(f"[V97] unet.attn_processors: {len(procs)}개, IPA 타입: {ipa_count}개", flush=True)
                    if ipa_count > 0:
                        sample_proc = next((v for v in procs.values() if 'IP' in type(v).__name__), None)
                        if sample_proc:
                            print(f"[V97] IPA proc type: {type(sample_proc).__name__}", flush=True)
                            # proc 내부 텐서 크기 확인
                            for attr_name in ['to_k_ip', 'to_v_ip', 'ip_layers', 'weight']:
                                attr_val = getattr(sample_proc, attr_name, None)
                                if attr_val is not None:
                                    if isinstance(attr_val, torch.Tensor):
                                        print(f"[V97]   proc.{attr_name}: shape={list(attr_val.shape)} device={attr_val.device}", flush=True)
                                    elif hasattr(attr_val, 'weight'):
                                        w = attr_val.weight
                                        print(f"[V97]   proc.{attr_name}.weight: shape={list(w.shape)} device={w.device}", flush=True)
                except Exception as pe:
                    print(f"[V97] attn_processors check error: {pe}", flush=True)

        except Exception as e:
            import traceback
            print(f"[V97] error: {e}\n{traceback.format_exc()}", flush=True)

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

        # V89: IPA job counter — 70개마다 worker 재시작 (503GB / 5.8GB ≈ 86개, 여유 16개)
        if mode == "ipadapter":
            global _ipa_job_count
            _ipa_job_count += 1
            print(f"[V89] IPA job count: {_ipa_job_count}", flush=True)
            if _ipa_job_count >= 20:
                print(f"[V90] Job limit reached ({_ipa_job_count}), restarting worker...", flush=True)
                import os
                os._exit(0)

        return {"image": image_b64, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        _force_vram_free()
        return {"error": str(e), "status": "failed"}


print("Registering handler...", flush=True)
runpod.serverless.start({"handler": handler})
