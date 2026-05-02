"""
[STEP 4] 멀티 에이전트 성능 평가 스크립트
python evaluate.py --checkpoint checkpoints/ep_03000.pt

평가 지표:
  - IoU      : Intersection over Union  (예측 ∩ GT) / (예측 ∪ GT)
  - Dice     : 2|P∩G| / (|P| + |G|)
  - Precision: 방문 픽셀 중 실제 혈관 비율  TP / (TP + FP)
  - Recall   : 전체 혈관 중 방문된 비율    TP / (TP + FN)
"""
import argparse
import os

import numpy as np
import cv2
import torch
from PIL import Image

from config import CONFIG
from src.agent import DQNAgent
from src.multi_agent import MultiAgentTracker
from src.dataset import build_registry


# ------------------------------------------------------------------
def compute_overlap_metrics(visited: np.ndarray, skeleton: np.ndarray) -> dict:
    """
    에이전트 방문맵(visited)과 GT 스켈레톤(skeleton) 간 Overlap 지표 계산.

    Parameters
    ----------
    visited  : (H, W) bool — 에이전트가 방문한 픽셀
    skeleton : (H, W) bool — Ground Truth 혈관 중심선

    Returns
    -------
    dict with keys: iou, dice, precision, recall, tp, fp, fn
    """
    pred = visited.astype(bool)
    gt   = skeleton.astype(bool)

    tp = int((pred &  gt).sum())   # 방문했고 혈관인 픽셀
    fp = int((pred & ~gt).sum())   # 방문했지만 혈관 아닌 픽셀
    fn = int((~pred & gt).sum())   # 혈관인데 방문 못 한 픽셀

    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice      = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = coverage

    return {
        "iou":       iou,
        "dice":      dice,
        "precision": precision,
        "recall":    recall,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ------------------------------------------------------------------
def render_overlap(image: np.ndarray, visited: np.ndarray, skeleton: np.ndarray,
                   tails: list | None = None) -> np.ndarray:
    """
    Overlap 시각화:
      초록(TP)  : 방문 + 혈관       → 정확히 추적
      빨강(FP)  : 방문 + 비혈관     → 혈관 이탈
      노랑(FN)  : 미방문 + 혈관     → 놓친 혈관
      마젠타(꼬리): 혈관 끝 이후 이탈 구간 (꼬리)
    """
    vis = (image * 255).astype(np.uint8).copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    pred = visited.astype(bool)
    gt   = skeleton.astype(bool)

    tp_mask = pred &  gt
    fp_mask = pred & ~gt
    fn_mask = ~pred & gt

    vis[fn_mask] = (0,   200, 200)  # 노랑 (BGR)
    vis[fp_mask] = (0,   0,   200)  # 빨강
    vis[tp_mask] = (0,   200, 0  )  # 초록

    # 꼬리 구간: 마젠타 (혈관 끝 이후 이탈)
    if tails:
        for tail in tails:
            for r, c in tail:
                vis[r, c] = (255, 0, 200)  # 마젠타 (BGR)

    return vis


# ------------------------------------------------------------------
def find_latest_checkpoint(checkpoints_dir: str) -> str:
    pts = sorted(
        f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")
    )
    if not pts:
        raise FileNotFoundError(f"체크포인트 없음: {checkpoints_dir}")
    return os.path.join(checkpoints_dir, pts[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="체크포인트 경로 (미지정 시 최고 에피소드 자동 선택)")
    parser.add_argument("--n_images",   type=int, default=20)
    args = parser.parse_args()

    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG["device"] = device

    ckpt_path = args.checkpoint or find_latest_checkpoint(CONFIG["checkpoints_dir"])
    print(f"체크포인트: {ckpt_path}")

    agent = DQNAgent(CONFIG)
    loaded_ep = agent.load(ckpt_path)
    print(f"사용 에피소드: ep {loaded_ep}")
    agent.epsilon = 0.0  # 그리디 정책

    tracker = MultiAgentTracker(agent)

    _, val_samples = build_registry(CONFIG, seed=CONFIG["random_seed"])
    n = min(args.n_images, len(val_samples))

    all_metrics = []

    for i in range(n):
        image, gray, mask, skeleton, _, branch_map, near_skel, endpoint_map, distance_map = val_samples[i]
        visited, trajectories, tails = tracker.track(
            image, mask, skeleton,
            branch_map=branch_map, near_skel=near_skel, endpoint_map=endpoint_map,
            distance_map=distance_map,
        )

        # ── Overlap 지표 (꼬리 제외된 visited 기준) ───────────────
        m = compute_overlap_metrics(visited, skeleton)

        all_steps  = sum(len(t) for t in trajectories)
        tail_steps = sum(len(t) for t in tails)
        n_agents   = len(trajectories)

        all_metrics.append({**m, "n_agents": n_agents,
                             "total_steps": all_steps, "tail_steps": tail_steps})

        # ── Overlap 시각화 저장 (꼬리 마젠타로 표시) ──────────────
        vis = render_overlap(image, visited, skeleton, tails=tails)
        cv2.imwrite(
            os.path.join(CONFIG["results_dir"], f"img_{i+1:02d}_overlap.png"),
            vis,
        )

        print(
            f"img {i+1:2d} | agents {n_agents:3d} | "
            f"IoU {m['iou']:.3f} | Dice {m['dice']:.3f} | "
            f"Prec {m['precision']:.3f} | Recall {m['recall']:.3f} | "
            f"TP {m['tp']:5d} FP {m['fp']:5d} FN {m['fn']:5d} | "
            f"tail {tail_steps:4d}px"
        )

    # ── 전체 평균 ─────────────────────────────────────────────────
    print("\n=== 평균 지표 ===")
    for key in ["iou", "dice", "precision", "recall", "n_agents", "total_steps", "tail_steps"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n결과 저장: {CONFIG['results_dir']}")


if __name__ == "__main__":
    main()
