"""
시각적 평가 스크립트.
val set 중 랜덤 10장을 추적하고, GT 기반 색상 비교 이미지를 저장.

색상 규칙 (검은 배경 위):
  파란색 (255,   0,   0) BGR — TP : GT 혈관이고 에이전트가 추적함
  빨간색 (  0,   0, 255) BGR — FP : GT 혈관 아닌데 에이전트가 추적함
  흰색   (255, 255, 255) BGR — FN : GT 혈관인데 에이전트가 가지 않음
  검정색 (  0,   0,   0) BGR — TN : GT 혈관도 아니고 에이전트도 안 감

실행:
    python visual_eval.py
    python visual_eval.py --checkpoint checkpoints/ep_29000.pt
    python visual_eval.py --n_images 10 --seed 0
"""
import argparse
import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch

from config import CONFIG
from src.agent import DQNAgent
from src.multi_agent import MultiAgentTracker
from src.dataset import build_registry
from evaluate import compute_overlap_metrics, find_latest_checkpoint


# ── 색상 상수 (BGR) ───────────────────────────────────────────────────
_TP   = (255,   0,   0)   # 파랑   — GT 혈관이고 방문함
_FP   = (  0,   0, 255)   # 빨강   — GT 혈관 아닌데 방문함
_FN   = (255, 255, 255)   # 흰색   — GT 혈관인데 방문 못 함
_TN   = (  0,   0,   0)   # 검정   — GT 혈관도 아니고 방문도 안 함
_TAIL = (255,   0, 200)   # 마젠타 — 혈관 끝 이후 이탈 구간(꼬리)


def make_visual(visited: np.ndarray, skeleton: np.ndarray,
                tails: list | None = None) -> np.ndarray:
    """
    visited(꼬리 제거된 방문맵)와 skeleton(GT)으로 색상 비교 이미지 생성.
    tails가 주어지면 꼬리 구간을 마젠타로 추가 표시.
    """
    H, W = skeleton.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)   # 전체 검정(TN)

    gt   = skeleton.astype(bool)
    pred = visited.astype(bool)

    canvas[gt  & ~pred] = _FN   # 흰색: FN
    canvas[gt  &  pred] = _TP   # 파랑: TP
    canvas[~gt &  pred] = _FP   # 빨강: FP

    # 꼬리 구간: 마젠타 (지표 계산에서 제외된 이탈 구간)
    if tails:
        for tail in tails:
            for r, c in tail:
                canvas[r, c] = _TAIL

    return canvas


def overlay_metrics(canvas: np.ndarray, m: dict, idx: int) -> np.ndarray:
    """이미지 좌상단에 지표 텍스트를 겹쳐 그린다."""
    vis = canvas.copy()
    lines = [
        f"img {idx:02d}",
        f"Dice  {m['dice']:.3f}",
        f"Prec  {m['precision']:.3f}",
        f"Recall{m['recall']:.3f}",
        f"IoU   {m['iou']:.3f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(vis, text, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="체크포인트 경로 (미지정 시 최고 에피소드 자동 선택)")
    parser.add_argument("--n_images",   type=int, default=10, help="시각화할 이미지 수")
    parser.add_argument("--seed",       type=int, default=None, help="랜덤 시드")
    args = parser.parse_args()

    # ── 출력 폴더 (results/visual_result/{YYYYMMDD_HHMMSS}/) ─────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(CONFIG["results_dir"], "visual_result", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"저장 폴더: {out_dir}")

    # ── 디바이스 ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG["device"] = device

    # ── 에이전트 로드 ─────────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_latest_checkpoint(CONFIG["checkpoints_dir"])
    print(f"체크포인트: {ckpt_path}")
    agent = DQNAgent(CONFIG)
    loaded_ep = agent.load(ckpt_path)
    print(f"사용 에피소드: ep {loaded_ep}")
    agent.epsilon = 0.0

    tracker = MultiAgentTracker(agent)

    # ── val set 로드 ──────────────────────────────────────────────────
    _, val_samples = build_registry(CONFIG, seed=CONFIG["random_seed"])
    n = min(args.n_images, len(val_samples))

    rng     = random.Random(args.seed)
    indices = rng.sample(range(len(val_samples)), n)
    print(f"val set {len(val_samples)}장 중 랜덤 {n}장 선택")

    # ── 추적 + 시각화 ─────────────────────────────────────────────────
    summary = []
    for rank, idx in enumerate(indices, start=1):
        image, _, mask, skeleton, _, branch_map, near_skel, endpoint_map = val_samples[idx]

        visited, _, tails = tracker.track(
            image, mask, skeleton,
            branch_map=branch_map, near_skel=near_skel, endpoint_map=endpoint_map,
        )

        m      = compute_overlap_metrics(visited, skeleton)
        canvas = make_visual(visited, skeleton, tails=tails)
        canvas = overlay_metrics(canvas, m, rank)

        save_path = os.path.join(out_dir, f"img_{rank:02d}.png")
        cv2.imwrite(save_path, canvas)
        summary.append(m)

        print(
            f"  img {rank:02d} (val[{idx:3d}]) | "
            f"Dice {m['dice']:.3f} | Prec {m['precision']:.3f} | "
            f"Recall {m['recall']:.3f} | IoU {m['iou']:.3f}"
        )

    # ── 평균 요약 ─────────────────────────────────────────────────────
    print("\n=== 평균 지표 ===")
    for key in ["iou", "dice", "precision", "recall"]:
        vals = [m[key] for m in summary]
        print(f"  {key:10s}: {np.mean(vals):.4f} +- {np.std(vals):.4f}")
    print(f"\n결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
