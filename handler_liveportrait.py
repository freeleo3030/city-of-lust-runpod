import runpod
import base64
import os
import sys
import gc
import tempfile
import numpy as np

print("LivePortrait handler starting...", flush=True)

WEIGHTS_DIR = "/runpod-volume/liveportrait"
APP_DIR = "/app"

# 표정별 기본 파라미터 (절차적 생성의 베이스값)
EXPRESSION_BASE = {
    #               smile  eyebrow  base_pitch  base_yaw  smile_var
    "idle":        (0.05,  0.0,     0.0,        0.0,      0.08),
    "smile":       (0.75,  0.15,   -1.5,       -2.0,      0.18),
    "shy":         (0.30, -0.15,    8.0,       -4.5,      0.12),
    "surprised":   (0.05,  0.50,   -3.5,        1.0,      0.08),
    "annoyed":     (-0.18,-0.40,    2.5,        4.5,      0.08),
    "disappointed":(-0.18,-0.30,    5.5,       -3.0,      0.08),
}

EXPRESSION_NAMES = set(EXPRESSION_BASE.keys())


def _smooth_interp(kps, t):
    """(t, v) 키포인트 리스트를 선형 보간"""
    for i in range(len(kps) - 1):
        t0, v0 = kps[i]
        t1, v1 = kps[i + 1]
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0 + 1e-8)
            return v0 * (1 - a) + v1 * a
    return kps[-1][1]


def generate_frame_params(expression, num_frames, fps=15):
    """절차적으로 자연스러운 프레임 파라미터 생성
    반환: list of (eyes_open, smile, eyebrow, pitch, yaw, gaze_x)
    """
    import math

    rng = np.random.default_rng()  # 매 호출마다 다른 랜덤
    duration = num_frames / fps
    dt = 1.0 / fps

    base = EXPRESSION_BASE.get(expression, EXPRESSION_BASE["idle"])
    base_smile, base_eyebrow, base_pitch, base_yaw, smile_var = base

    # ── 눈 깜빡임 스케줄 (2~5초 랜덤 간격, 20% 확률로 더블 블링크) ──
    blink_times = []
    t = rng.uniform(0.4, 1.8)
    while t < duration - 0.15:
        blink_times.append(t)
        if rng.random() < 0.22:           # double blink
            blink_times.append(t + 0.16)
        t += rng.uniform(2.0, 5.0)
    blink_dur = 0.13  # 초

    # ── 고개 움직임 웨이포인트 ──
    yaw_kps   = [(0.0, base_yaw)]
    pitch_kps = [(0.0, base_pitch)]

    t = rng.uniform(0.6, 1.4)
    while t < duration:
        yaw_kps.append((t,   base_yaw   + rng.uniform(-5.5, 5.5)))
        pitch_kps.append((t, base_pitch + rng.uniform(-3.5, 3.5)))
        t += rng.uniform(0.9, 2.2)
    yaw_kps.append((duration, base_yaw))
    pitch_kps.append((duration, base_pitch))

    # ── 시선 이동 (gaze_x: -0.02~0.02 수평 오프셋) ──
    gaze_kps = [(0.0, 0.0)]
    t = rng.uniform(1.0, 2.5)
    while t < duration:
        gaze_kps.append((t, rng.uniform(-0.018, 0.018)))
        t += rng.uniform(1.5, 3.5)
    gaze_kps.append((duration, 0.0))

    # ── 미소 변화 (기본값 주변을 ±smile_var 로 부드럽게 변동) ──
    smile_kps = [(0.0, base_smile)]
    t = rng.uniform(0.5, 1.5)
    while t < duration:
        smile_kps.append((t, np.clip(base_smile + rng.uniform(-smile_var, smile_var), -0.3, 1.0)))
        t += rng.uniform(0.7, 1.8)
    smile_kps.append((duration, base_smile))

    # ── 프레임별 파라미터 생성 ──
    result = []
    for i in range(num_frames):
        t = i * dt

        # 눈 열림 값 (블링크 중이면 sin 곡선으로 닫힘)
        eyes_open = 1.0
        for bt in blink_times:
            if bt <= t <= bt + blink_dur:
                progress = (t - bt) / blink_dur
                eyes_open = max(0.03, abs(math.sin(progress * math.pi)))
                break

        yaw    = _smooth_interp(yaw_kps, t)
        pitch  = _smooth_interp(pitch_kps, t)
        smile  = _smooth_interp(smile_kps, t)
        gaze_x = _smooth_interp(gaze_kps, t)

        result.append((eyes_open, float(smile), float(base_eyebrow),
                       float(pitch), float(yaw), float(gaze_x)))
    return result

loaded_pipeline = None


def download_weights():
    """LivePortrait 모델 가중치 다운로드 (최초 1회)"""
    marker = os.path.join(WEIGHTS_DIR, ".downloaded")
    if os.path.exists(marker):
        print("LivePortrait weights already present.", flush=True)
        return

    print("Downloading LivePortrait weights from HuggingFace...", flush=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=WEIGHTS_DIR,
        ignore_patterns=["*.md", "*.txt", "examples/*"],
    )

    open(marker, "w").close()
    print("Weights downloaded!", flush=True)


def load_pipeline():
    global loaded_pipeline
    if loaded_pipeline is not None:
        return

    download_weights()

    # symlink: /app/pretrained_weights/ 구조 항상 보장 (download 스킵돼도 실행)
    pretrained_dir = os.path.join(APP_DIR, "pretrained_weights")
    os.makedirs(pretrained_dir, exist_ok=True)
    for folder in ["liveportrait", "insightface"]:
        link = os.path.join(pretrained_dir, folder)
        src = os.path.join(WEIGHTS_DIR, folder)
        if os.path.exists(src) and not os.path.exists(link):
            os.symlink(src, link)
            print(f"Symlinked {src} -> {link}", flush=True)

    sys.path.insert(0, APP_DIR)

    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig

    # 실제 볼륨 구조: /runpod-volume/liveportrait/liveportrait/base_models/*.pth
    base_dir = os.path.join(WEIGHTS_DIR, "liveportrait", "base_models")
    retarget_dir = os.path.join(WEIGHTS_DIR, "liveportrait", "retargeting_models")
    inference_cfg = InferenceConfig(
        checkpoint_F=os.path.join(base_dir, "appearance_feature_extractor.pth"),
        checkpoint_M=os.path.join(base_dir, "motion_extractor.pth"),
        checkpoint_W=os.path.join(base_dir, "warping_module.pth"),
        checkpoint_G=os.path.join(base_dir, "spade_generator.pth"),
        checkpoint_S=os.path.join(retarget_dir, "stitching_retargeting_module.pth"),
    )
    crop_cfg = CropConfig()

    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.cropper import Cropper

    wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
    cropper = Cropper(crop_cfg=crop_cfg)

    loaded_pipeline = {"wrapper": wrapper, "cropper": cropper, "crop_cfg": crop_cfg}
    print("LivePortrait wrapper loaded!", flush=True)


def generate_animation(source_image_b64, expression, num_frames=30, fps=15):
    """소스 이미지 + 표정 → mp4 base64"""
    from PIL import Image
    from io import BytesIO
    import imageio
    import torch
    import math

    # 소스 이미지 디코딩
    if "," in source_image_b64:
        source_image_b64 = source_image_b64.split(",", 1)[1]
    raw = source_image_b64.strip()
    raw += "=" * (-len(raw) % 4)
    pil_src = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
    img_rgb = np.array(pil_src)

    wrapper = loaded_pipeline["wrapper"]
    cropper = loaded_pipeline["cropper"]
    crop_cfg = loaded_pipeline["crop_cfg"]

    # 소스 이미지 crop
    crop_info = cropper.crop_source_image(img_rgb, crop_cfg)
    img_crop_256x256 = crop_info["img_crop_256x256"]  # (256,256,3) uint8

    # 절차적 프레임 파라미터 생성
    frame_params = generate_frame_params(expression, num_frames, fps)

    # 소스 특징 추출 및 프레임 생성
    with wrapper.inference_ctx():
        I_s = wrapper.prepare_source(img_crop_256x256)
        x_s_info = wrapper.get_kp_info(I_s)
        f_s = wrapper.extract_feature_3d(I_s)
        x_s = wrapper.transform_keypoint(x_s_info)

        output_frames = []
        for eyes_open, smile, eyebrow, pitch_delta, yaw_delta, gaze_x in frame_params:
            x_d_i = x_s.clone()

            # 고개 회전 (pitch/yaw)
            offset = torch.zeros_like(x_d_i)
            offset[..., 0] += math.radians(yaw_delta) * 0.08
            offset[..., 1] += math.radians(pitch_delta) * 0.08
            x_d_i = x_d_i + offset

            # 시선 이동 (눈 관련 kp만 x축 이동)
            if abs(gaze_x) > 0.001:
                x_d_i[:, 2:8, 0] += gaze_x

            # 눈 깜빡임 (눈 kp y축 조정)
            eye_close = max(0.0, 1.0 - eyes_open)
            if eye_close > 0.05:
                x_d_i[:, 2:8, 1] += eye_close * 0.022

            # 미소 (입 아래쪽 kp y축 조정: 11~15번 추정)
            if abs(smile) > 0.02:
                x_d_i[:, 11:16, 1] -= smile * 0.012   # 음수=아래로 당김=미소

            x_d_i = wrapper.stitching(x_s, x_d_i)
            out = wrapper.warp_decode(f_s, x_s, x_d_i)
            frame = wrapper.parse_output(out["out"])[0]  # (1,H,W,3) -> (H,W,3)
            output_frames.append(frame)

    # 루프를 위해 역방향 프레임 추가 (forward + backward = seamless loop)
    loop_frames = output_frames + output_frames[::-1]

    # mp4 인코딩
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = imageio.get_writer(tmp.name, fps=fps, codec="libx264", quality=8, macro_block_size=2)
    for frame in loop_frames:
        writer.append_data(frame)
    writer.close()

    with open(tmp.name, "rb") as f:
        mp4_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp.name)

    gc.collect()
    torch.cuda.empty_cache()

    return mp4_b64


def handler(job):
    try:
        inp = job["input"]
        mode = inp.get("mode", "liveportrait")

        if mode == "list_dir":
            # 디버그: 볼륨 디렉토리 구조 출력
            import glob
            files = []
            for root, dirs, fnames in os.walk(WEIGHTS_DIR):
                for f in fnames:
                    files.append(os.path.join(root, f).replace(WEIGHTS_DIR, ""))
            return {"files": sorted(files)[:100], "weights_dir": WEIGHTS_DIR, "status": "success"}

        if mode != "liveportrait":
            return {"error": "This endpoint only supports mode=liveportrait", "status": "failed"}

        source_image = inp.get("source_image", "")
        if not source_image:
            return {"error": "source_image (base64) required", "status": "failed"}

        expression = inp.get("expression", "idle")
        if expression not in EXPRESSION_NAMES:
            expression = "idle"

        num_frames = int(inp.get("num_frames", 30))   # 30프레임 @ 15fps = 2초 루프
        fps = int(inp.get("fps", 15))

        print(f"Generating expression={expression}, {num_frames}frames @ {fps}fps", flush=True)

        load_pipeline()

        mp4_b64 = generate_animation(source_image, expression, num_frames, fps)

        return {"video": mp4_b64, "expression": expression, "status": "success"}

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return {"error": str(e), "status": "failed"}


print("Registering LivePortrait handler...", flush=True)
runpod.serverless.start({"handler": handler})
