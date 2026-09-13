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

# 표정별 애니메이션 파라미터 정의
# eyes_open: 1.0=정상, 0.0=완전히 감김
# smile: 0.0~1.0
# eyebrow_raise: -1.0~1.0
# head_pitch: 도(°), 양수=아래, 음수=위
# head_yaw: 도(°), 양수=오른쪽, 음수=왼쪽
EXPRESSION_KEYFRAMES = {
    "idle": [
        # (frame_ratio, eyes_open, smile, eyebrow_raise, head_pitch, head_yaw)
        (0.0,  1.0,  0.05, 0.0,   0.0,  0.0),
        (0.15, 0.9,  0.05, 0.0,   0.5,  0.5),
        (0.3,  0.05, 0.05, 0.0,   1.0,  1.0),   # blink
        (0.4,  1.0,  0.05, 0.0,   0.5,  0.5),
        (0.6,  1.0,  0.05, 0.0,  -0.5, -0.5),
        (0.75, 0.9,  0.05, 0.0,   0.0,  0.0),
        (0.85, 0.05, 0.05, 0.0,   0.0,  0.0),   # blink
        (1.0,  1.0,  0.05, 0.0,   0.0,  0.0),
    ],
    "smile": [
        (0.0,  1.0,  0.0,  0.0,  0.0,  0.0),
        (0.2,  1.0,  0.4,  0.1,  0.0, -1.0),
        (0.4,  0.9,  0.7,  0.2, -1.0, -2.0),
        (0.6,  1.0,  0.9,  0.2, -1.0, -1.0),
        (0.8,  0.05, 0.8,  0.1,  0.0,  0.0),    # blink while smiling
        (1.0,  1.0,  0.7,  0.1,  0.0,  0.0),
    ],
    "shy": [
        (0.0,  1.0,  0.0,  0.0,   0.0,  0.0),
        (0.2,  0.8,  0.2, -0.1,   5.0, -3.0),
        (0.4,  0.7,  0.4, -0.2,  10.0, -5.0),   # 고개 숙임
        (0.6,  0.05, 0.3, -0.1,   8.0, -4.0),   # blink + 아래 봄
        (0.8,  0.6,  0.3, -0.1,   8.0, -3.0),
        (1.0,  0.8,  0.2,  0.0,   5.0, -2.0),
    ],
    "surprised": [
        (0.0,  1.0,  0.0,  0.0,  0.0,  0.0),
        (0.1,  1.3,  0.0,  0.5, -3.0,  0.0),   # 눈 크게 + 눈썹 올림
        (0.3,  1.4,  0.0,  0.6, -4.0,  2.0),
        (0.5,  1.3,  0.1,  0.5, -3.0,  1.0),
        (0.7,  1.2,  0.1,  0.3, -2.0,  0.0),
        (0.9,  0.05, 0.1,  0.2, -1.0,  0.0),   # blink
        (1.0,  1.1,  0.0,  0.1,  0.0,  0.0),
    ],
    "annoyed": [
        (0.0,  1.0,  0.0,  0.0,  0.0,  0.0),
        (0.2,  0.8, -0.1, -0.3,  2.0,  3.0),   # 눈썹 찡그림
        (0.4,  0.7, -0.2, -0.4,  3.0,  5.0),
        (0.6,  0.05,-0.1, -0.3,  2.0,  3.0),   # blink
        (0.8,  0.8, -0.1, -0.3,  2.0,  2.0),
        (1.0,  0.9,  0.0, -0.1,  1.0,  1.0),
    ],
    "disappointed": [
        (0.0,  1.0,  0.0,  0.0,  0.0,  0.0),
        (0.2,  0.9, -0.1, -0.2,  3.0, -2.0),
        (0.4,  0.8, -0.2, -0.3,  6.0, -4.0),   # 고개 아래
        (0.6,  0.05,-0.2, -0.2,  5.0, -3.0),   # blink
        (0.8,  0.7, -0.2, -0.2,  5.0, -2.0),
        (1.0,  0.8, -0.1, -0.1,  3.0, -1.0),
    ],
}

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


def interpolate_keyframes(keyframes, num_frames):
    """keyframe 리스트를 num_frames 개수의 프레임으로 보간"""
    result = []
    kf = np.array(keyframes)  # shape: (N, 6)  [ratio, eyes, smile, eyebrow, pitch, yaw]

    for i in range(num_frames):
        t = i / (num_frames - 1)
        # t에 해당하는 구간 찾기
        for j in range(len(kf) - 1):
            t0, t1 = kf[j, 0], kf[j + 1, 0]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0 + 1e-8)
                vals = kf[j, 1:] * (1 - alpha) + kf[j + 1, 1:] * alpha
                result.append(vals)
                break
        else:
            result.append(kf[-1, 1:])

    return result  # list of [eyes_open, smile, eyebrow, pitch, yaw]


def generate_animation(source_image_b64, expression, num_frames=30, fps=15):
    """소스 이미지 + 표정 → mp4 base64"""
    from PIL import Image
    from io import BytesIO
    import imageio
    import torch

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

    # 소스 특징 추출
    with wrapper.inference_ctx():
        I_s = wrapper.prepare_source(img_crop_256x256)         # (1,3,256,256)
        x_s_info = wrapper.get_kp_info(I_s)                    # pitch/yaw/roll/kp/exp/scale/t
        x_c_s = x_s_info["kp"]                                 # canonical keypoints
        f_s = wrapper.extract_feature_3d(I_s)                  # appearance feature
        x_s = wrapper.transform_keypoint(x_s_info)             # source keypoints

        # 소스 눈/입 기준 비율 (landmark 기반)
        source_lmk = crop_info.get("lmk_crop")

        keyframes = EXPRESSION_KEYFRAMES.get(expression, EXPRESSION_KEYFRAMES["idle"])
        frame_params = interpolate_keyframes(keyframes, num_frames)

        import math
        output_frames = []
        for eyes_open, smile, eyebrow, pitch_delta, yaw_delta in frame_params:
            # keypoint에 직접 offset 적용 (단순화된 표정 제어)
            x_d_i = x_s.clone()

            # pitch/yaw offset (고개 움직임)
            if abs(pitch_delta) > 0.01 or abs(yaw_delta) > 0.01:
                offset = torch.zeros_like(x_d_i)
                offset[..., 1] += math.radians(float(pitch_delta)) * 0.08
                offset[..., 0] += math.radians(float(yaw_delta)) * 0.08
                x_d_i = x_d_i + offset

            # 눈 감기: 눈 관련 keypoints y축 조정 (21개 중 눈 영역 추정 2~8번)
            eye_close = max(0.0, 1.0 - float(eyes_open))
            if eye_close > 0.05:
                x_d_i[:, 2:8, 1] += eye_close * 0.02

            # stitching
            x_d_i = wrapper.stitching(x_s, x_d_i)

            # warp & decode
            out = wrapper.warp_decode(f_s, x_s, x_d_i)
            frame = wrapper.parse_output(out["out"])   # (H,W,3) uint8
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
        if expression not in EXPRESSION_KEYFRAMES:
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
