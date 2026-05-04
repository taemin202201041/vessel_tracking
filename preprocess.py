"""
[STEP 1] GT → 스켈레톤 변환

GT를 그대로 1픽셀 폭 중심선으로 변환. 형태 조작 없음.

blood_vessel/*.png (462장) + blood_vessel/*.gif (DRIVE 20장)
→ E:\\...\\skeletons\\*_skeleton.png

실행:
    python preprocess.py            # 새 파일만
    python preprocess.py -rebuild  # 전체 재생성
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from config import CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="이미 생성된 스켈레톤도 재생성")
    args = parser.parse_args()

    gt_dir   = CONFIG["new_gt_dir"]
    out_dir  = CONFIG["new_skeletons_dir"]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(gt_dir)
        if f.lower().endswith(".png") or f.lower().endswith(".gif")
    )
    print(f"GT 파일 수: {len(files)} (PNG + GIF)")

    done = skipped = 0
    for i, fname in enumerate(files):
        stem      = os.path.splitext(fname)[0]
        skel_path = os.path.join(out_dir, f"{stem}_skeleton.png")

        gt        = np.array(Image.open(os.path.join(gt_dir, fname)).convert("L"))
        binary_u8 = (gt > 16).astype(np.uint8)


        if os.path.exists(skel_path) and not args.rebuild:
            skipped += 1
            continue

        # 반복 침식: 매 스텝마다 dt 재계산 → 혈관 소실 방지
        result = binary_u8.copy()
        while True:
            dt     = cv2.distanceTransform(result, cv2.DIST_L2, 5)
            max_dt = cv2.dilate(dt, np.ones((5, 5), np.uint8))
            target = (dt == 1) & (max_dt >= 4)
            if target.sum() == 0:
                break
            result[target] = 0

        # 고립 픽셀 제거: 8이웃이 모두 0인 픽셀 → 0
        kernel = np.ones((3, 3), np.float32)
        neighbor_sum = cv2.filter2D(result.astype(np.float32), -1, kernel) - result
        result[(result == 1) & (neighbor_sum == 0)] = 0

        skel = skeletonize(result > 0).astype(np.uint8) * 255
        Image.fromarray(skel).save(skel_path)
        done += 1

        if done % 20 == 1 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {fname} → {stem}_skeleton.png")

    print(f"\n완료. 신규 생성 {done}개 / 건너뜀 {skipped}개")
    print(f"저장 위치: {out_dir}")


if __name__ == "__main__":
    main()