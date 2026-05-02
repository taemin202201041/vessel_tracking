"""
혈관 유틸리티: optic disc 감지, 분기점 추출, 방향 계산.
"""
import os
import numpy as np
import cv2
import torch

# ── Disc Segmentation 모델 lazy singleton ────────────────────────────
_disc_model  = None
_disc_device = None


def _get_disc_model():
    global _disc_model, _disc_device
    if _disc_model is not None:
        return _disc_model, _disc_device
    try:
        from config import CONFIG
        ckpt = CONFIG.get("disc_seg_checkpoint", "")
        if ckpt and os.path.exists(ckpt):
            from src.disc_model import DiscSegNet
            _disc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _disc_model  = DiscSegNet().to(_disc_device)
            _disc_model.load_state_dict(
                torch.load(ckpt, map_location=_disc_device, weights_only=True)
            )
            _disc_model.eval()
            print(f"[DiscSeg] 모델 로드: {ckpt}")
    except Exception as e:
        print(f"[DiscSeg] 모델 로드 실패 (HoughCircles fallback): {e}")
    return _disc_model, _disc_device

# 8방향 이동 벡터 (environment.py와 동일)
DIRECTIONS = [
    (-1,  0), (-1,  1), (0,  1), (1,  1),
    ( 1,  0), ( 1, -1), (0, -1), (-1, -1),
]


def find_branch_points(skeleton):
    """스켈레톤에서 분기점 추출 — Crossing Number 알고리즘.
    시계방향 0→1 전환 횟수(CN) >= 3인 픽셀만 진짜 분기점으로 판정.
    단순 이웃 카운팅은 곡선 세그먼트를 분기점으로 오인하는 문제가 있음."""
    skel = skeleton.astype(np.uint8)
    padded = np.pad(skel, 1, mode='constant', constant_values=0)

    # 시계방향 8이웃: P2(상), P3(우상), P4(우), P5(우하), P6(하), P7(좌하), P8(좌), P9(좌상)
    neighbors = [
        padded[:-2, 1:-1],   # P2 상
        padded[:-2, 2:],     # P3 우상
        padded[1:-1, 2:],    # P4 우
        padded[2:, 2:],      # P5 우하
        padded[2:, 1:-1],    # P6 하
        padded[2:, :-2],     # P7 좌하
        padded[1:-1, :-2],   # P8 좌
        padded[:-2, :-2],    # P9 좌상
    ]

    nb = np.stack(neighbors)                                    # (8, H, W)
    cn = ((nb == 0) & (np.roll(nb, -1, axis=0) == 1)).sum(axis=0).astype(np.uint8)

    return (skel > 0) & (cn >= 3)


def find_endpoints(skeleton):
    """스켈레톤 끝점 추출 — 이웃 스켈레톤 픽셀이 정확히 1개인 픽셀."""
    skel = skeleton.astype(np.uint8)
    padded = np.pad(skel, 1, mode='constant', constant_values=0)
    neighbors = [
        padded[:-2, 1:-1], padded[:-2, 2:],  padded[1:-1, 2:],  padded[2:, 2:],
        padded[2:, 1:-1],  padded[2:, :-2],  padded[1:-1, :-2], padded[:-2, :-2],
    ]
    nb = np.stack(neighbors)
    neighbor_count = nb.sum(axis=0)
    return (skel > 0) & (neighbor_count == 1)


def _disc_radius(H, W):
    """이미지 단변의 7% — 해상도에 무관하게 disc 반경 추정."""
    return max(20, int(min(H, W) * 0.07))


def find_optic_disc_center(image, mask):
    """
    Optic disc 중심 찾기.
    1순위: DL 모델 (disc_seg.pt 존재 시 자동 로드)
    2순위: HoughCircles
    3순위: FOV 내 red 채널 최대값
    """
    # ── 1순위: DL 모델 ─────────────────────────────────────
    model, device = _get_disc_model()
    if model is not None:
        from src.disc_model import predict_disc_center
        result = predict_disc_center(model, image, mask, device)
        if result is not None:
            return result

    # ── 2순위: HoughCircles ────────────────────────────────
    H, W = image.shape[:2]
    disc_r_px = _disc_radius(H, W)
    fov = mask.astype(np.uint8)

    if image.dtype != np.uint8:
        red_u8 = (np.clip(image[:, :, 0], 0, 1) * 255).astype(np.uint8)
    else:
        red_u8 = image[:, :, 0].copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(red_u8)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=fov)
    blurred  = cv2.GaussianBlur(enhanced, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(50, disc_r_px),
        param1=50,
        param2=25,
        minRadius=max(10, disc_r_px - 15),
        maxRadius=disc_r_px + 20,
    )

    if circles is not None:
        x, y, _ = circles[0][0]
        return (int(round(y)), int(round(x)))

    # ── 3순위: red 채널 최대값 ─────────────────────────────
    masked = enhanced.astype(np.float32) * fov
    idx = np.argmax(masked)
    r, c = np.unravel_index(idx, masked.shape)
    return (int(r), int(c))


def find_start_in_optic_disc(image, mask, skeleton=None):
    """
    optic disc 안에서 최적 시작점 찾기.

    학습 시 (skeleton 있음):
      - 이미지에서 disc 중심 찾기
      - disc 안 green 채널 어두운 픽셀 = 혈관 후보
      - 스켈레톤(GT)으로 실제 혈관 위 픽셀 검증
      - 스켈레톤 위에 있는 어두운 픽셀 중 가장 어두운 점 반환

    테스트 시 (skeleton=None):
      - 동일하게 disc 안 가장 어두운 픽셀 반환
      - 학습 때 검증된 상관관계(어두움=혈관)로 신뢰 가능
    """
    disc_r, disc_c = find_optic_disc_center(image, mask)
    H, W = image.shape[:2]
    disc_radius = _disc_radius(H, W)

    # disc 반경 내 좌표 생성
    rs = np.arange(max(0, disc_r - disc_radius), min(H, disc_r + disc_radius + 1))
    cs = np.arange(max(0, disc_c - disc_radius), min(W, disc_c + disc_radius + 1))
    rr, cc = np.meshgrid(rs, cs, indexing='ij')
    in_disc = (rr - disc_r) ** 2 + (cc - disc_c) ** 2 <= disc_radius ** 2
    in_fov = mask[rr, cc]
    valid = in_disc & in_fov

    if not valid.any():
        return (disc_r, disc_c)

    # green 채널 어두운 픽셀 = 혈관 후보 (혈관은 green에서 어둡게 나타남)
    green = image[:, :, 1]
    green_vals = green[rr[valid], cc[valid]]
    candidate_rs = rr[valid]
    candidate_cs = cc[valid]

    if skeleton is not None:
        # GT 검증: 스켈레톤 위에 있는 후보만 유지
        on_skel = skeleton[candidate_rs, candidate_cs]
        if on_skel.any():
            candidate_rs = candidate_rs[on_skel]
            candidate_cs = candidate_cs[on_skel]
            # disc 중심에 가장 가까운 스켈레톤 픽셀 반환 (어두운 픽셀보다 중심에 가까움)
            dists = (candidate_rs - disc_r) ** 2 + (candidate_cs - disc_c) ** 2
            closest_idx = int(np.argmin(dists))
            return (int(candidate_rs[closest_idx]), int(candidate_cs[closest_idx]))

    # skeleton 없을 때: 가장 어두운 픽셀 (혈관 추정)
    darkest_idx = int(np.argmin(green_vals))
    return (int(candidate_rs[darkest_idx]), int(candidate_cs[darkest_idx]))


def find_optic_disc_from_image(image, mask):
    """하위 호환용."""
    return find_optic_disc_center(image, mask)


def get_skeleton_directions(skeleton, r, c, came_from_dir=None):
    """(r,c)에서 스켈레톤으로 이어지는 방향들 반환. came_from 방향(역방향) 제외."""
    H, W = skeleton.shape
    dirs = []
    for d, (dr, dc) in enumerate(DIRECTIONS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and skeleton[nr, nc] and d != came_from_dir:
            dirs.append(d)
    return dirs
