"""
체크포인트별 성능 비교 벤치마크.
ep_01000 ~ ep_29000 (2000 단위, 총 15개) 각각 evaluate 실행.

결과 구조:
  results/benchmark/
    ep_01000/   ← 이미지 20장 overlap 시각화
    ep_03000/
    ...
    ep_29000/
    summary.csv ← 전체 지표 비교표
"""
import os
import csv

import numpy as np
import torch
import cv2

from config import CONFIG
from src.environment import RetinalVesselEnv
from src.agent import DQNAgent
from src.multi_agent import MultiAgentTracker
from evaluate import compute_overlap_metrics, render_overlap

CHECKPOINTS = [f"ep_{ep:05d}" for ep in range(1000, 30000, 2000)]  # 15개
BENCHMARK_DIR = os.path.join(CONFIG["results_dir"], "benchmark")


def run_one(ckpt_name, samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG["device"] = device

    agent = DQNAgent(CONFIG)
    agent.load(os.path.join(CONFIG["checkpoints_dir"], f"{ckpt_name}.pt"))
    agent.epsilon = 0.0

    tracker = MultiAgentTracker(agent)
    metrics_list = []

    for i, (image, gray, mask, skeleton, _, branch_map, near_skel, endpoint_map) in enumerate(samples):
        visited, trajectories, tails = tracker.track(
            image, mask, skeleton,
            branch_map=branch_map, near_skel=near_skel, endpoint_map=endpoint_map,
        )

        m = compute_overlap_metrics(visited, skeleton)
        n_agents  = len(trajectories)
        all_steps = sum(len(t) for t in trajectories)
        metrics_list.append({**m, "n_agents": n_agents, "total_steps": all_steps})

        vis = render_overlap(image, visited, skeleton, tails=tails)
        cv2.imwrite(os.path.join(out_dir, f"img_{i+1:02d}.png"), vis)

    avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
    return avg


def main():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    env     = RetinalVesselEnv(split="train")
    samples = env.samples  # 20장 전체

    summary = []

    for ckpt_name in CHECKPOINTS:
        ckpt_path = os.path.join(CONFIG["checkpoints_dir"], f"{ckpt_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {ckpt_name} — 파일 없음")
            continue

        out_dir = os.path.join(BENCHMARK_DIR, ckpt_name)
        print(f"\n[{ckpt_name}] 평가 중...")

        avg = run_one(ckpt_name, samples, out_dir)
        summary.append({"checkpoint": ckpt_name, **avg})

        print(
            f"  IoU {avg['iou']:.4f} | Dice {avg['dice']:.4f} | "
            f"Prec {avg['precision']:.4f} | Recall {avg['recall']:.4f} | "
            f"Agents {avg['n_agents']:.1f}"
        )

    # summary.csv 저장
    csv_path = os.path.join(BENCHMARK_DIR, "summary.csv")
    fieldnames = ["checkpoint", "iou", "dice", "precision", "recall",
                  "tp", "fp", "fn", "n_agents", "total_steps"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n=== 벤치마크 완료 ===")
    print(f"요약: {csv_path}")
    print(f"\n{'checkpoint':>12} | {'IoU':>6} | {'Dice':>6} | {'Prec':>6} | {'Recall':>6}")
    print("-" * 52)
    for row in summary:
        print(f"{row['checkpoint']:>12} | {row['iou']:>6.4f} | {row['dice']:>6.4f} | "
              f"{row['precision']:>6.4f} | {row['recall']:>6.4f}")


if __name__ == "__main__":
    main()
