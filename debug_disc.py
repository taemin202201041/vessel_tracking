"""
시신경 유두(optic disc) 감지 디버깅 스크립트.
훈련 이미지 전체에 감지 결과를 시각화해서 results/disc_debug/ 에 저장.
"""
import os
import cv2
import numpy as np
from PIL import Image as PILImage
from config import CONFIG
from src.vessel_utils import find_optic_disc_center

OUT_DIR = os.path.join(CONFIG["results_dir"], "disc_debug")
os.makedirs(OUT_DIR, exist_ok=True)

img_dir  = CONFIG["train_images_dir"]
mask_dir = CONFIG["train_masks_dir"]

img_files  = sorted(f for f in os.listdir(img_dir)  if f.endswith(".tif"))
mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".gif"))

for img_f, mask_f in zip(img_files, mask_files):
    img_path  = os.path.join(img_dir,  img_f)
    mask_path = os.path.join(mask_dir, mask_f)

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask    = np.array(PILImage.open(mask_path).convert("L")) > 127

    r, c = find_optic_disc_center(img_rgb, mask)

    # 감지 결과 시각화 (원 + 중심점)
    vis = img_bgr.copy()
    cv2.circle(vis, (c, r), 40, (0, 255, 0), 2)   # 초록 원 (disc 반경)
    cv2.circle(vis, (c, r), 4,  (0, 0, 255), -1)  # 빨간 점 (중심)
    cv2.putText(vis, f"({r},{c})", (c + 10, r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out_path = os.path.join(OUT_DIR, img_f.replace(".tif", "_disc.png"))
    cv2.imwrite(out_path, vis)
    print(f"{img_f}: disc center = ({r}, {c})")

print(f"\n결과 저장 완료: {OUT_DIR}")
