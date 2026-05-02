"""
[STEP 1] GT → 스켈레톤 변환

GT를 그대로 1픽셀 폭 중심선으로 변환. 형태 조작 없음.

blood_vessel/*.png (462장) + blood_vessel/*.gif (DRIVE 20장)
→ E:\\...\\skeletons\\*_skeleton.png

실행:
    python preprocess.py            # 새 파일만
    python preprocess.py --rebuild  # 전체 재생성
"""
import argparse
import os

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from config import CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="이미 생성된 스켈레톤도 재생성")
    args = parser.parse_args()

    gt_dir  = CONFIG["new_gt_dir"]
    out_dir = CONFIG["new_skeletons_dir"]
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

        if os.path.exists(skel_path) and not args.rebuild:
            skipped += 1
            continue

        gt     = np.array(Image.open(os.path.join(gt_dir, fname)).convert("L"))
        binary = gt > 127
        skel   = skeletonize(binary).astype(np.uint8) * 255
        Image.fromarray(skel).save(skel_path)
        done += 1

        if done % 20 == 1 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {fname} → {stem}_skeleton.png")

    print(f"\n완료. 신규 생성 {done}개 / 건너뜀 {skipped}개")
    print(f"저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
