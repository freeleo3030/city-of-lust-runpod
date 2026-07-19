"""
포즈 skeleton PNG 생성 스크립트
실행: python generate_skeletons.py  (runpod/ 또는 runpod/skeletons/ 어디서든)
결과: runpod/skeletons/ 폴더에 PNG 4장 생성
      → Supabase Storage char-images/skeletons/ 에 업로드
"""
from PIL import Image, ImageDraw
import os

W, H = 384, 512

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255),
    (255, 0, 85), (255, 0, 170),
    (255, 0, 255), (170, 0, 255),
]

# idx: 0=nose,1=neck,2=r_sho,3=r_elb,4=r_wri,5=l_sho,6=l_elb,7=l_wri
#       8=r_hip,9=r_kne,10=r_ank,11=l_hip,12=l_kne,13=l_ank
#       14=r_eye,15=l_eye,16=r_ear,17=l_ear

POSES = {
    # 정상위 (missionary.png)
    # 위에서 내려다본 누운 자세: 다리를 위로 들어올린 역V자
    # 무릎 y < 골반 y → 다리가 위로 들림 → M자 퍼짐 제거 → SD 단일인물 인식
    "missionary": [
        (192, 155),  # 0 nose
        (192, 190),  # 1 neck
        (148, 215),  # 2 r_shoulder
        (132, 168),  # 3 r_elbow
        (150, 122),  # 4 r_wrist  ← 손목 머리 위
        (236, 215),  # 5 l_shoulder
        (252, 168),  # 6 l_elbow
        (234, 122),  # 7 l_wrist
        (162, 395),  # 8 r_hip
        (120, 305),  # 9 r_knee   ← 다리 위로 들림 (y < hip)
        (148, 230),  # 10 r_ankle ← 발목 어깨 높이
        (222, 395),  # 11 l_hip
        (264, 305),  # 12 l_knee
        (236, 230),  # 13 l_ankle
        (180, 145),  # 14 r_eye
        (204, 145),  # 15 l_eye
        (170, 155),  # 16 r_ear
        (214, 155),  # 17 l_ear
    ],
    # 후배위 (doggy.png)
    # 카메라: 엉덩이 정면 뒤에서 좌우 대칭
    # 종아리는 무릎에서 약간 바깥쪽으로 땅에 붙어있음
    "doggy": [
        (192, 118),  # 0 nose
        (192, 148),  # 1 neck
        (150, 165),  # 2 r_shoulder
        (144, 258),  # 3 r_elbow
        (140, 352),  # 4 r_wrist
        (234, 165),  # 5 l_shoulder
        (240, 258),  # 6 l_elbow
        (244, 352),  # 7 l_wrist
        (142, 262),  # 8 r_hip
        (122, 375),  # 9 r_knee
        (87,  455),  # 10 r_ankle ← 무릎보다 바깥
        (242, 262),  # 11 l_hip
        (262, 375),  # 12 l_knee
        (297, 455),  # 13 l_ankle
        (182, 108),  # 14 r_eye
        (202, 108),  # 15 l_eye
        (170, 116),  # 16 r_ear
        (214, 116),  # 17 l_ear
    ],
    # 여성상위 (cowgirl.png)
    # 카메라: 정면에서 직립 앉은 자세
    # 발목: 무릎보다 바깥, 무릎~허리 사이 높이
    "cowgirl": [
        (192, 62),   # 0 nose
        (192, 100),  # 1 neck
        (150, 120),  # 2 r_shoulder
        (125, 195),  # 3 r_elbow
        (112, 268),  # 4 r_wrist
        (234, 120),  # 5 l_shoulder
        (259, 195),  # 6 l_elbow
        (272, 268),  # 7 l_wrist
        (162, 268),  # 8 r_hip
        (128, 390),  # 9 r_knee
        (92,  335),  # 10 r_ankle ← 무릎 바깥, 무릎~허리 사이
        (222, 268),  # 11 l_hip
        (256, 390),  # 12 l_knee
        (292, 335),  # 13 l_ankle
        (182, 52),   # 14 r_eye
        (202, 52),   # 15 l_eye
        (170, 60),   # 16 r_ear
        (214, 60),   # 17 l_ear
    ],
    # 버터플라이 (butterfly.png)
    # 카메라: 정면, 앉아서 다리 옆으로 벌림, 손은 등 뒤로
    # 무릎이 허리 위에 위치
    "butterfly": [
        (192, 72),   # 0 nose
        (192, 108),  # 1 neck
        (148, 130),  # 2 r_shoulder
        (175, 215),  # 3 r_elbow (등 뒤로)
        (192, 305),  # 4 r_wrist
        (236, 130),  # 5 l_shoulder
        (209, 215),  # 6 l_elbow
        (192, 305),  # 7 l_wrist
        (158, 295),  # 8 r_hip
        (85,  215),  # 9 r_knee  ← 무릎이 허리 위
        (42,  330),  # 10 r_ankle
        (226, 295),  # 11 l_hip
        (299, 215),  # 12 l_knee
        (342, 330),  # 13 l_ankle
        (182, 62),   # 14 r_eye
        (202, 62),   # 15 l_eye
        (170, 70),   # 16 r_ear
        (214, 70),   # 17 l_ear
    ],
}


def draw_skeleton(keypoints, path):
    img = Image.new('RGB', (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, (a, b) in enumerate(CONNECTIONS):
        if keypoints[a] and keypoints[b]:
            d.line([keypoints[a], keypoints[b]], fill=COLORS[i], width=5)
    for pt in keypoints:
        if pt:
            x, y = pt
            d.ellipse([x-6, y-6, x+6, y+6], fill=(255, 255, 255))
    img.save(path)
    print(f"생성: {path}")


# 이 스크립트가 있는 폴더 기준으로 skeletons/ 서브폴더에 저장
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skeletons")
os.makedirs(OUT, exist_ok=True)

for pose_name, kps in POSES.items():
    draw_skeleton(kps, os.path.join(OUT, f"{pose_name}.png"))

print(f"\n완료! {OUT} 의 PNG 4장을 Supabase Storage > char-images/skeletons/ 에 업로드하세요.")
