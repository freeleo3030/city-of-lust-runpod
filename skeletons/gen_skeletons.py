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
    print(f"Saved: {path}")

OUT = os.path.dirname(__file__)

# idx: 0=nose,1=neck,2=r_sho,3=r_elb,4=r_wri,5=l_sho,6=l_elb,7=l_wri
#       8=r_hip,9=r_kne,10=r_ank,11=l_hip,12=l_kne,13=l_ank
#       14=r_eye,15=l_eye,16=r_ear,17=l_ear

# ─── 1. MISSIONARY (missionary.png) ─────────────────────────────────────
# 위에서 내려다본 누운 자세: 다리를 위로 들어올린 역V자
# 핵심: 무릎 y < 골반 y (다리가 위로 올라감) → M자 퍼짐 제거 → SD 단일인물 인식
missionary = [
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
]
draw_skeleton(missionary, os.path.join(OUT, 'missionary.png'))

# ─── 2. DOGGY (doggy.png) ─────────────────────────────────────────────
# 정면 뒤에서 본 후배위: 엉덩이가 중앙, 다리 아래로, 팔은 앞으로 뻗음
# 좌우 대칭 (카메라가 엉덩이 정면 뒤에서 바라봄)
doggy = [
    (192, 118),  # 0 nose  (머리 멀리 위에)
    (192, 148),  # 1 neck
    (150, 165),  # 2 r_shoulder
    (144, 258),  # 3 r_elbow (팔 앞으로 내려감)
    (140, 352),  # 4 r_wrist
    (234, 165),  # 5 l_shoulder
    (240, 258),  # 6 l_elbow
    (244, 352),  # 7 l_wrist
    (142, 262),  # 8 r_hip
    (122, 375),  # 9 r_knee
    (87,  455),  # 10 r_ankle (10도 더 바깥으로)
    (242, 262),  # 11 l_hip
    (262, 375),  # 12 l_knee
    (297, 455),  # 13 l_ankle
    (182, 108),  # 14 r_eye
    (202, 108),  # 15 l_eye
    (170, 116),  # 16 r_ear
    (214, 116),  # 17 l_ear
]
draw_skeleton(doggy, os.path.join(OUT, 'doggy.png'))

# ─── 3. COWGIRL (cowgirl.png) ─────────────────────────────────────────
# 여성 상위: 직립으로 앉아있는 자세, 무릎 꿇고 넓게 벌림
cowgirl = [
    (192, 62),   # 0 nose
    (192, 100),  # 1 neck
    (150, 120),  # 2 r_shoulder
    (125, 195),  # 3 r_elbow
    (112, 268),  # 4 r_wrist
    (234, 120),  # 5 l_shoulder
    (259, 195),  # 6 l_elbow
    (272, 268),  # 7 l_wrist
    (162, 268),  # 8 r_hip
    (128, 390),  # 9 r_knee (넓게 벌림)
    (92,  335),  # 10 r_ankle (무릎 바깥, 무릎~허리 사이)
    (222, 268),  # 11 l_hip
    (256, 390),  # 12 l_knee
    (292, 335),  # 13 l_ankle (무릎 바깥, 무릎~허리 사이)
    (182, 52),   # 14 r_eye
    (202, 52),   # 15 l_eye
    (170, 60),   # 16 r_ear
    (214, 60),   # 17 l_ear
]
draw_skeleton(cowgirl, os.path.join(OUT, 'cowgirl.png'))

# ─── 4. SIDE (butterfly.png) ──────────────────────────────────────────────
# 정면 앉은 자세, 다리 옆으로 벌림, 팔은 등 뒤로 짚음, 무릎이 허리 위
side = [
    (192, 72),   # 0 nose
    (192, 108),  # 1 neck
    (148, 130),  # 2 r_shoulder
    (175, 215),  # 3 r_elbow (등 뒤로)
    (192, 305),  # 4 r_wrist
    (236, 130),  # 5 l_shoulder
    (209, 215),  # 6 l_elbow
    (192, 305),  # 7 l_wrist
    (158, 295),  # 8 r_hip
    (85,  215),  # 9 r_knee  (무릎이 허리 위)
    (42,  330),  # 10 r_ankle (종아리 바깥 아래)
    (226, 295),  # 11 l_hip
    (299, 215),  # 12 l_knee
    (342, 330),  # 13 l_ankle
    (182, 62),   # 14 r_eye
    (202, 62),   # 15 l_eye
    (170, 70),   # 16 r_ear
    (214, 70),   # 17 l_ear
]
draw_skeleton(side, os.path.join(OUT, 'butterfly.png'))

print("Done!")
