"""
포즈 skeleton PNG 생성 스크립트
실행: python generate_skeletons.py
결과: skeletons/ 폴더에 PNG 4장 생성 → Supabase Storage에 업로드
"""
from PIL import Image, ImageDraw
import os

W, H = 768, 1024

# keypoint index: 0=nose,1=neck,2=rshoulder,3=relbow,4=rwrist,
#                 5=lshoulder,6=lelbow,7=lwrist,
#                 8=rhip,9=rknee,10=rankle,
#                 11=lhip,12=lknee,13=lankle,
#                 14=reye,15=leye,16=rear,17=lear
LIMBS = [
    (1, 0), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17),
]
LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 170), (0, 255, 255), (0, 170, 255),
    (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (255, 0, 170), (255, 0, 85), (200, 0, 200), (150, 0, 255),
]
KP_COLOR = (255, 255, 255)

POSES = {
    # 정상위: 등 대고 누움, 다리 넓게 벌려 들어올림, 카메라 위에서 내려다봄
    # 팔을 옆으로 크게 벌림(누운 자세), 다리는 아래쪽으로 벌어져 내려감
    "missionary": [
        (384, 95),    # 0 nose
        (384, 185),   # 1 neck
        (510, 230),   # 2 rshoulder  ← 어깨 넓게
        (600, 385),   # 3 relbow     ← 팔 바깥으로 크게 벌림
        (640, 520),   # 4 rwrist
        (260, 230),   # 5 lshoulder
        (180, 385),   # 6 lelbow
        (140, 520),   # 7 lwrist
        (460, 555),   # 8 rhip
        (565, 730),   # 9 rknee  ← 다리 벌어져 아래로
        (610, 880),   # 10 rankle
        (310, 555),   # 11 lhip
        (205, 730),   # 12 lknee
        (160, 880),   # 13 lankle
        (420, 72),    # 14 reye
        (348, 72),    # 15 leye
        (452, 84),    # 16 rear
        (316, 84),    # 17 lear
    ],
    # 후배위: 네발 엎드림, 카메라 뒤 낮은 위치에서
    # 엉덩이가 화면 하단(카메라 가까운 쪽), 머리·팔이 상단(멀리)
    "doggy": [
        (384, 80),    # 0 nose  ← 머리 상단에 작게
        (384, 165),   # 1 neck
        (480, 215),   # 2 rshoulder  ← 어깨 상단(앞쪽으로 뻗음)
        (530, 390),   # 3 relbow
        (535, 555),   # 4 rwrist  ← 손이 침대에 짚음
        (290, 215),   # 5 lshoulder
        (245, 390),   # 6 lelbow
        (240, 555),   # 7 lwrist
        (465, 700),   # 8 rhip  ← 엉덩이 하단 중앙(카메라 가까움)
        (510, 875),   # 9 rknee
        (545, 975),   # 10 rankle
        (305, 700),   # 11 lhip
        (250, 875),   # 12 lknee
        (215, 975),   # 13 lankle
        (420, 60),    # 14 reye
        (348, 60),    # 15 leye
        (455, 72),    # 16 rear
        (313, 72),    # 17 lear
    ],
    # 여성상위: 올라탄 자세, 카메라 아래서 올려다봄
    # 가슴이 화면 중앙 크게, 다리가 아래쪽으로 넓게 벌어짐
    "cowgirl": [
        (384, 65),    # 0 nose  ← 얼굴 상단 작게
        (384, 155),   # 1 neck
        (490, 200),   # 2 rshoulder
        (545, 370),   # 3 relbow
        (525, 530),   # 4 rwrist  ← 허벅지에 손
        (280, 200),   # 5 lshoulder
        (230, 370),   # 6 lelbow
        (250, 530),   # 7 lwrist
        (460, 625),   # 8 rhip
        (580, 830),   # 9 rknee  ← 무릎 양쪽으로 크게 벌어짐
        (650, 965),   # 10 rankle
        (310, 625),   # 11 lhip
        (190, 830),   # 12 lknee
        (110, 965),   # 13 lankle
        (418, 45),    # 14 reye
        (350, 45),    # 15 leye
        (455, 57),    # 16 rear
        (313, 57),    # 17 lear
    ],
    # 좌위: 앉아서 나비자세, 무릎이 바깥으로 크게 벌어지고 발목은 안쪽
    "side": [
        (384, 90),    # 0 nose
        (384, 185),   # 1 neck
        (495, 225),   # 2 rshoulder
        (550, 385),   # 3 relbow
        (530, 545),   # 4 rwrist  ← 허벅지 안쪽
        (275, 225),   # 5 lshoulder
        (220, 385),   # 6 lelbow
        (240, 545),   # 7 lwrist
        (465, 580),   # 8 rhip
        (625, 745),   # 9 rknee  ← 무릎 옆으로 크게
        (570, 655),   # 10 rankle ← 발목은 안쪽(나비자세)
        (305, 580),   # 11 lhip
        (148, 745),   # 12 lknee
        (210, 655),   # 13 lankle
        (422, 67),    # 14 reye
        (346, 67),    # 15 leye
        (455, 79),    # 16 rear
        (313, 79),    # 17 lear
    ],
}


def draw_skeleton(keypoints):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 팔다리 연결선
    for idx, (a, b) in enumerate(LIMBS):
        if a < len(keypoints) and b < len(keypoints):
            color = LIMB_COLORS[idx] if idx < len(LIMB_COLORS) else (128, 128, 128)
            draw.line([keypoints[a], keypoints[b]], fill=color, width=8)

    # 관절 점
    for kp in keypoints:
        r = 10
        draw.ellipse([kp[0]-r, kp[1]-r, kp[0]+r, kp[1]+r], fill=KP_COLOR)

    return img


os.makedirs("skeletons", exist_ok=True)

for pose_name, kps in POSES.items():
    img = draw_skeleton(kps)
    path = f"skeletons/{pose_name}.png"
    img.save(path)
    print(f"생성: {path}")

print("\n완료! skeletons/ 폴더의 PNG 4장을 Supabase Storage > pose-skeletons 버킷에 업로드하세요.")
