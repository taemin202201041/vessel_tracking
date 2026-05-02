"""
통합 데이터셋 로더.

소스:
  1. 기존 DRIVE 20장  — data/train/images/*.tif + data/train/skeletons/*.png
  2. 신규 PNG 462장   — new_images_dir/*.png    + new_skeletons_dir/*_skeleton.png

분할:
  - seed=42로 전체 482쌍을 섞은 뒤 8:2 train/val 고정 분할
  - SampleList.__getitem__ 호출 시 on-demand 로드 (메모리 절약)
"""
import os
import random

import cv2
import numpy as np
from PIL import Image as PILImage

from src.vessel_utils import (
    find_start_in_optic_disc,
    find_branch_points,
    find_endpoints,
)


# ── 조명 보정 (environment.py와 동일) ────────────────────────────────
def _correct_illumination(img, mask):
    fov = mask.astype(np.float32)
    corrected = np.empty_like(img)
    for ch in range(3):
        channel = img[:, :, ch]
        bg = cv2.GaussianBlur(channel, (101, 101), 30)
        bg = np.where(bg < 1e-6, 1e-6, bg)
        mean_val = float(channel[mask].mean()) if mask.any() else 1.0
        corrected[:, :, ch] = np.clip(channel * (mean_val / bg) * fov, 0, 1)
    return corrected


# ── FOV 마스크 자동 생성 (명시적 마스크 없을 때) ──────────────────────
def _make_fov_mask(img):
    """Green 채널 밝기 기반으로 원형 FOV 영역을 감지해 bool 마스크 반환."""
    gray_u8 = (img[:, :, 1] * 255).astype(np.uint8)
    _, m = cv2.threshold(gray_u8, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
    return m > 0


# ── 단일 샘플 로드 ────────────────────────────────────────────────────
def _load_one(meta: dict, patch_size: int, near_skel_radius: int = 1):
    """
    meta 딕셔너리에서 경로를 읽어 학습용 튜플을 반환.
    반환: (img, gray_padded, mask, skel, start_pos, branch_map, near_skel, endpoint_map)
    """
    h = patch_size // 2

    # 이미지 로드
    img = cv2.imread(meta["img_path"], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {meta['img_path']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # FOV 마스크
    if meta.get("mask_path") and os.path.exists(meta["mask_path"]):
        mask = np.array(PILImage.open(meta["mask_path"]).convert("L")) > 127
    else:
        mask = _make_fov_mask(img)

    # 조명 보정 + green 채널
    img = _correct_illumination(img, mask)
    gray = img[:, :, 1]
    gray_padded = np.pad(gray, h, mode="constant", constant_values=0.0)

    # 스켈레톤
    skel = np.array(PILImage.open(meta["skel_path"]).convert("L")) > 127

    # 파생 맵
    branch_map  = find_branch_points(skel)
    k = near_skel_radius * 2 + 1
    near_skel   = cv2.dilate(skel.astype(np.uint8),
                             np.ones((k, k), np.uint8)).astype(bool)
    endpoint_map = find_endpoints(skel)
    start_pos   = find_start_in_optic_disc(img, mask, skeleton=skel)

    return (img, gray_padded, mask, skel, start_pos,
            branch_map, near_skel, endpoint_map)


# ── Lazy-loading + 캐시 컨테이너 ─────────────────────────────────────
class SampleList:
    """
    metas 리스트를 감싸 __getitem__ 호출 시 on-demand로 샘플을 로드한다.
    한 번 로드한 샘플은 메모리에 캐시해 재사용 (매 에피소드 디스크 I/O 제거).
    """
    def __init__(self, metas: list, patch_size: int, near_skel_radius: int = 1):
        self.metas            = metas
        self.patch_size       = patch_size
        self.near_skel_radius = near_skel_radius
        self._cache: dict     = {}

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = _load_one(self.metas[idx], self.patch_size, self.near_skel_radius)
        return self._cache[idx]

    def random_sample(self):
        return self[random.randrange(len(self))]


# ── 메타 수집 ─────────────────────────────────────────────────────────
def _collect_new_metas(cfg) -> list:
    """
    vessel_test/ 이미지 전체 (PNG 462장 + DRIVE tif 20장) 메타 수집.

    매칭 규칙:
      PNG : vessel_test/X.png      ↔ blood_vessel/X.png         (동일 파일명)
      TIF : vessel_test/N_training.tif ↔ blood_vessel/N_manual1.gif
    스켈레톤:
      PNG : skeletons/X_skeleton.png
      GIF : skeletons/N_manual1_skeleton.png  (GT stem 기준)
    """
    img_dir  = cfg["new_images_dir"]
    gt_dir   = cfg["new_gt_dir"]
    skel_dir = cfg["new_skeletons_dir"]

    if not os.path.isdir(skel_dir):
        print(f"  [WARNING] 스켈레톤 폴더 없음: {skel_dir}\n"
              "  → python preprocess.py 를 먼저 실행하세요.")
        return []

    metas = []
    for img_f in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(img_f)[1].lower()

        if ext == ".png":
            stem      = os.path.splitext(img_f)[0]
            gt_path   = os.path.join(gt_dir, img_f)          # 동일 파일명
            skel_path = os.path.join(skel_dir, f"{stem}_skeleton.png")
            source    = stem.split("-")[0]                    # "FIVES" / "HRF" / ...

        elif ext == ".tif":
            # "21_training.tif" → GT: "21_manual1.gif", skel: "21_manual1_skeleton.png"
            num       = os.path.splitext(img_f)[0].split("_")[0]
            gt_fname  = f"{num}_manual1.gif"
            gt_path   = os.path.join(gt_dir, gt_fname)
            skel_path = os.path.join(skel_dir, f"{num}_manual1_skeleton.png")
            source    = "DRIVE"

        else:
            continue

        if not os.path.exists(gt_path) or not os.path.exists(skel_path):
            continue

        metas.append({
            "img_path":  os.path.join(img_dir, img_f),
            "gt_path":   gt_path,
            "skel_path": skel_path,
            "mask_path": None,   # FOV 마스크 자동 생성
            "source":    source,
        })
    return metas


# ── 공개 API ─────────────────────────────────────────────────────────
def build_registry(cfg, seed: int = 42):
    """
    전체 metas를 수집해 train/val SampleList로 분할 반환.

    Returns
    -------
    train_samples : SampleList
    val_samples   : SampleList
    """
    patch_size = cfg["patch_size"]
    val_ratio  = cfg.get("val_ratio", 0.2)

    metas = _collect_new_metas(cfg)

    if not metas:
        raise RuntimeError("로드 가능한 샘플이 없습니다. preprocess*.py를 먼저 실행하세요.")

    rng     = np.random.RandomState(seed)
    indices = rng.permutation(len(metas)).tolist()
    n_val   = max(1, int(len(metas) * val_ratio))

    val_metas   = [metas[i] for i in indices[:n_val]]
    train_metas = [metas[i] for i in indices[n_val:]]

    near_skel_radius = cfg.get("near_skel_radius", 1)

    print(f"데이터셋 구성: 전체 {len(metas)}장 → train {len(train_metas)} / val {len(val_metas)}")
    src_counts = {}
    for m in metas:
        src_counts[m["source"]] = src_counts.get(m["source"], 0) + 1
    for src, cnt in sorted(src_counts.items()):
        print(f"  {src:10s}: {cnt}장")

    return (SampleList(train_metas, patch_size, near_skel_radius),
            SampleList(val_metas,   patch_size, near_skel_radius))
