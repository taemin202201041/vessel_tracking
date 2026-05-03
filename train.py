"""
[STEP 3] 학습 메인 스크립트 (멀티 에이전트 기반)
python train.py
python train.py --resume checkpoints/ep_01000.pt
python train.py --resume checkpoints/ep_10000.pt --render
python train.py --resume checkpoints/ep_01000.pt --reset_epsilon --render
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CONFIG
from src.agent import DQNAgent
from src.multi_agent import _Agent, MultiAgentTracker, trim_tail
from src.vessel_utils import DIRECTIONS, get_skeleton_directions
from src.dataset import build_registry
from evaluate import compute_overlap_metrics

LEARN_FREQ = 8   # batch_size 128에 맞춰 조정 (동일 데이터 효율, GPU 활용률 향상)
MAX_REVISIT = 10
RENDER_EVERY = 30   # N step마다 1회 렌더 (높을수록 빠름, 낮을수록 부드러움)
WINDOW_NAME = "Vessel Tracking"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# BGR 색상 상수 (visual_eval과 동일)
_C_TP    = (255,   0,   0)   # 파랑   — TP: GT 혈관이고 방문
_C_FP    = (  0,   0, 255)   # 빨강   — FP: GT 혈관 아닌데 방문
_C_FN    = (255, 255, 255)   # 흰색   — FN: GT 혈관인데 미방문
_C_TAIL  = (255,   0, 200)   # 마젠타 — 꼬리 구간
_C_CUR   = (  0, 255, 255)   # 노랑   — 현재 위치
_C_START = (  0, 255,   0)   # 초록   — 시작점


def render_frame_ma(image, active_agents, finished_trajs, ep, step, n_agents,
                    start_pos=None, skeleton=None, visited=None):
    """
    멀티 에이전트 실시간 시각화.
    GT 스켈레톤 기준으로 TP/FP/FN/꼬리 색상 구분.
    """
    vis = (image * 255).astype(np.uint8).copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    if skeleton is not None and visited is not None:
        # FN: GT 혈관인데 아직 방문 안 한 픽셀
        fn_mask = skeleton & ~visited
        vis[fn_mask] = _C_FN

        # 완료된 궤적: trim_tail로 꼬리 분리 후 TP/FP/꼬리 표시
        for traj in finished_trajs:
            if not traj:
                continue
            vessel, tail = trim_tail(traj, skeleton)
            for r, c in vessel:
                vis[r, c] = _C_TP if skeleton[r, c] else _C_FP
            for r, c in tail:
                vis[r, c] = _C_TAIL

        # 활성 에이전트 궤적: TP/FP
        for ag in active_agents:
            for r, c in ag.trajectory:
                vis[r, c] = _C_TP if skeleton[r, c] else _C_FP
            r, c = ag.pos
            cv2.circle(vis, (c, r), 3, _C_CUR, -1)
    else:
        for traj in finished_trajs:
            for r, c in traj:
                vis[r, c] = (200, 200, 200)
        for ag in active_agents:
            for r, c in ag.trajectory:
                vis[r, c] = (255, 255, 255)
            r, c = ag.pos
            cv2.circle(vis, (c, r), 3, _C_CUR, -1)

    if start_pos is not None:
        sr, sc = start_pos
        cv2.circle(vis, (sc, sr), 6, _C_START, -1)

    cv2.putText(vis, f"ep {ep}  step {step}  agents {n_agents}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 창 크기에 맞게 nearest-neighbor로 리사이즈 (보간으로 인한 색 섞임 방지)
    H_v, W_v = vis.shape[:2]
    scale = min(1.0, 800 / max(H_v, W_v))
    if scale < 1.0:
        vis = cv2.resize(vis, (int(W_v * scale), int(H_v * scale)),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(WINDOW_NAME, vis)
    cv2.waitKey(1)


# ------------------------------------------------------------------
def run_ma_episode(dqn_agent, sample, cfg, total_steps_ref, ep=0, do_render=False):
    """
    멀티 에이전트 에피소드 실행.
    분기점마다 에이전트 스폰, 모든 transition을 replay buffer에 저장.
    반환: (ep_reward, avg_loss, ep_steps, all_trajectories, image)
    """
    image, gray_padded, mask, skeleton, optic_disc, branch_map, near_skel, endpoint_map, distance_map = sample
    H, W = skeleton.shape
    h = cfg["patch_size"] // 2
    p = cfg["patch_size"]
    n_actions = cfg["n_actions"]
    max_steps = cfg["max_steps"]
    max_off_vessel = cfg.get("max_off_vessel", 20)
    max_agents = cfg.get("max_agents", 200)
    step_reward = cfg.get("reward_step", 0.0)
    near_r = cfg.get("near_skel_radius", 2)

    visited = np.zeros((H, W), dtype=bool)
    visited_padded = np.zeros((H + 2 * h, W + 2 * h), dtype=bool)
    processed_branches = set()  # 이미 처리한 분기점 (중복 트리거 방지)

    # 방향별 dir_map 미리 계산 (np.full 반복 제거)
    dir_maps = [np.full((p, p), d / (n_actions - 1), dtype=np.float32) for d in range(n_actions)]

    # obs_batch 미리 할당 (매 step 재할당 제거)
    obs_buf = np.empty((max_agents, 3, p, p), dtype=np.float32)

    # 시작점: optic disc
    r0, c0 = optic_disc
    if r0 < h or r0 >= H - h or c0 < h or c0 >= W - h or not mask[r0, c0]:
        pts = np.argwhere(skeleton)
        valid = pts[(pts[:, 0] >= h) & (pts[:, 0] < H - h) &
                    (pts[:, 1] >= h) & (pts[:, 1] < W - h)]
        if len(valid) == 0:
            valid = pts
        chosen = valid[random.randrange(len(valid))]
        r0, c0 = int(chosen[0]), int(chosen[1])
    visited[r0, c0] = True
    visited_padded[r0 + h, c0 + h] = True

    start_dirs = get_skeleton_directions(skeleton, r0, c0)

    def get_obs(pos, last_dir, out):
        r, c = pos
        out[0] = gray_padded[r:r + p, c:c + p]
        out[1] = dir_maps[last_dir]
        out[2] = visited_padded[r:r + p, c:c + p]
        return out

    # 시작점을 분기점처럼 취급: 스켈레톤이 뻗는 모든 방향으로 에이전트 뿌리기
    active = []
    for d in (start_dirs if start_dirs else [0]):
        sdr, sdc = DIRECTIONS[d]
        snr, snc = r0 + sdr, c0 + sdc
        if 0 <= snr < H and 0 <= snc < W and mask[snr, snc] and not visited[snr, snc]:
            ag = _Agent((snr, snc), last_dir=d)
            ag.trajectory = [(r0, c0), (snr, snc)]
            visited[snr, snc] = True
            visited_padded[snr + h, snc + h] = True
            active.append(ag)
    if not active:
        initial_dir = start_dirs[0] if start_dirs else 0
        active = [_Agent((r0, c0), last_dir=initial_dir)]
    all_trajectories = []

    ep_reward = 0.0
    ep_loss = 0.0
    ep_q = 0.0
    loss_count = 0
    ep_steps = 0
    ep_spawned = 0

    while active:
        n = len(active)
        # 미리 할당된 버퍼에 직접 쓰기 (재할당 없음)
        for i, ag in enumerate(active):
            get_obs(ag.pos, ag.last_dir, obs_buf[i])
        obs_batch = obs_buf[:n]

        # epsilon-greedy 배치 행동 선택 (전진 편향: 50% 확률로 후진 방향 제외)
        actions = dqn_agent.select_actions_batch(obs_batch)
        for i in range(n):
            if random.random() < dqn_agent.epsilon:
                if random.random() < 0.5:
                    back = (active[i].last_dir + 4) % n_actions
                    choices = [a for a in range(n_actions) if a != back]
                    actions[i] = random.choice(choices)
                else:
                    actions[i] = random.randrange(n_actions)

        next_active = []
        spawned = []

        for i, (ag, action) in enumerate(zip(active, actions)):
            obs = obs_buf[i]
            r, c = ag.pos
            dr, dc = DIRECTIONS[action]
            nr, nc = r + dr, c + dc
            ag.steps += 1
            ag.last_dir = action

            # 경계 / FOV 이탈
            if not (0 <= nr < H and 0 <= nc < W) or not mask[nr, nc]:
                dqn_agent.store(obs, action, cfg["reward_boundary"], obs, 1.0)
                all_trajectories.append(ag.trajectory)
                continue

            # 재방문
            if visited[nr, nc]:
                ag.revisit_streak += 1
                ag.pos = (nr, nc)
                # 최근 window 내 재방문만 패널티 — 오래된 위치 복귀(분기 탐색 후 교차로 복귀 등)는 허용
                reward = cfg["reward_revisit"]
                done = ag.revisit_streak >= MAX_REVISIT or ag.steps >= max_steps
                next_obs = np.empty((3, p, p), dtype=np.float32)
                get_obs(ag.pos, ag.last_dir, next_obs)
                dqn_agent.store(obs, action, reward, next_obs, float(done))
                ep_reward += reward
                if done:
                    all_trajectories.append(ag.trajectory)
                else:
                    next_active.append(ag)
                continue

            # 정상 이동
            ag.revisit_streak = 0
            ag.pos = (nr, nc)
            visited[nr, nc] = True
            visited_padded[nr + h, nc + h] = True
            ag.trajectory.append((nr, nc))

            # 거리 기반 연속 보상 (Paper 2 r1 단순화)
            dist = float(distance_map[nr, nc])
            if dist == 0:
                reward = cfg["reward_on_vessel"] + step_reward
                ag.off_vessel_streak = 0
            elif dist <= near_r:
                reward = cfg["reward_near_vessel"] * (1.0 - dist / (near_r + 1)) + step_reward
                ag.off_vessel_streak = 0
            else:
                penalty = min(dist * 0.1, 2.0)
                reward = cfg["reward_off_vessel"] - penalty + step_reward
                ag.off_vessel_streak += 1

            # 혈관 끝점 도달 → 즉시 종료
            if endpoint_map[nr, nc]:
                next_obs = np.empty((3, p, p), dtype=np.float32)
                get_obs(ag.pos, ag.last_dir, next_obs)
                dqn_agent.store(obs, action, reward, next_obs, 1.0)
                ep_reward += reward
                all_trajectories.append(ag.trajectory)
                continue

            # 혈관 완전 이탈 연속 N스텝 → 종료
            if ag.off_vessel_streak >= max_off_vessel:
                next_obs = np.empty((3, p, p), dtype=np.float32)
                get_obs(ag.pos, ag.last_dir, next_obs)
                dqn_agent.store(obs, action, cfg["reward_boundary"], next_obs, 1.0)
                all_trajectories.append(ag.trajectory)
                continue

            done = ag.steps >= max_steps
            next_obs = np.empty((3, p, p), dtype=np.float32)
            get_obs(ag.pos, ag.last_dir, next_obs)
            dqn_agent.store(obs, action, reward, next_obs, float(done))
            ep_reward += reward

            # 분기점: 현재 위치 또는 인접 스켈레톤 픽셀이 분기점이면 스폰
            # (에이전트가 near_vessel에 있어도 감지 가능)
            branch_pos = None
            if branch_map[nr, nc]:
                branch_pos = (nr, nc)
            else:
                # 반경 3픽셀 내 분기점 탐색 — numpy 슬라이싱으로 벡터화
                r0b, r1b = max(0, nr-7), min(H, nr+8)
                c0b, c1b = max(0, nc-7), min(W, nc+8)
                sub = branch_map[r0b:r1b, c0b:c1b]
                if sub.any():
                    pts = np.argwhere(sub)
                    dists = (pts[:,0] - (nr-r0b))**2 + (pts[:,1] - (nc-c0b))**2
                    best = pts[np.argmin(dists)]
                    branch_pos = (r0b + int(best[0]), c0b + int(best[1]))

            if branch_pos is not None and branch_pos not in processed_branches:
                processed_branches.add(branch_pos)
                br, bc = branch_pos
                # 에이전트 위치와 분기점이 다를 수 있으므로 분기점 기준으로 came_from 재계산
                if (nr, nc) == (br, bc):
                    came_from = (action + 4) % 8
                else:
                    dr, dc = nr - br, nc - bc
                    came_from = min(range(len(DIRECTIONS)),
                                    key=lambda d: (DIRECTIONS[d][0] - dr) ** 2 + (DIRECTIONS[d][1] - dc) ** 2)
                branch_dirs = get_skeleton_directions(skeleton, br, bc, came_from_dir=came_from)
                n_spawned_here = 0
                for bd in branch_dirs:
                    if len(next_active) + len(spawned) >= max_agents:
                        break
                    bdr, bdc = DIRECTIONS[bd]
                    bnr, bnc = br + bdr, bc + bdc
                    if 0 <= bnr < H and 0 <= bnc < W and mask[bnr, bnc] and not visited[bnr, bnc]:
                        new_ag = _Agent((bnr, bnc), last_dir=bd, is_spawned=True)
                        new_ag.trajectory = [(br, bc), (bnr, bnc)]
                        visited[bnr, bnc] = True
                        visited_padded[bnr + h, bnc + h] = True
                        spawned.append(new_ag)
                        ep_spawned += 1
                        n_spawned_here += 1
                # 실제로 스폰된 경우에만 원래 에이전트 종료
                if n_spawned_here > 0:
                    all_trajectories.append(ag.trajectory)
                    continue

            if not done:
                next_active.append(ag)
            else:
                all_trajectories.append(ag.trajectory)

        active = (next_active + spawned)[:max_agents]
        ep_steps += 1

        # 학습
        total_steps_ref[0] += 1
        if total_steps_ref[0] % LEARN_FREQ == 0:
            result = dqn_agent.learn()
            if result is not None:
                loss, avg_q = result
                ep_loss += loss
                ep_q += avg_q
                loss_count += 1

        # 렌더링 (RENDER_EVERY step마다 1회 — 매 step 렌더 시 병목)
        if do_render and ep_steps % RENDER_EVERY == 0:
            render_frame_ma(image, active, all_trajectories, ep, ep_steps, len(active),
                            start_pos=(r0, c0), skeleton=skeleton, visited=visited)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return ep_reward, ep_loss / max(loss_count, 1), ep_q / max(loss_count, 1), ep_steps, ep_spawned, all_trajectories, image, True

    # 에피소드 종료 후 커버리지 패널티: ep_reward 로그에만 반영 (dummy transition 제거)
    skel_total = int(skeleton.sum())
    if skel_total > 0:
        visited_on_skel = int((visited & skeleton).sum())
        missed_ratio = 1.0 - visited_on_skel / skel_total
        if missed_ratio > 0:
            coverage_penalty = missed_ratio * cfg.get("reward_coverage_penalty", 30.0)
            ep_reward -= coverage_penalty

    avg_loss = ep_loss / loss_count if loss_count > 0 else 0.0
    avg_q    = ep_q    / loss_count if loss_count > 0 else 0.0
    return ep_reward, avg_loss, avg_q, ep_steps, ep_spawned, all_trajectories, image, False


# ------------------------------------------------------------------
# ------------------------------------------------------------------
def save_learning_curve(reward_history, q_history, val_history, results_dir, window=50):
    """
    results/learning_curve.png 저장.
    subplot 1: Average Return, subplot 2: Average Q, subplot 3: Val Dice
    """
    os.makedirs(results_dir, exist_ok=True)

    def smooth(arr, w):
        return [np.mean(arr[max(0, i-w):i+1]) for i in range(len(arr))]

    n_plots = 3 if val_history else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=False)

    eps_train = np.arange(1, len(reward_history) + 1)
    axes[0].plot(eps_train, smooth(reward_history, window), color="steelblue", linewidth=1.5)
    axes[0].set_ylabel(f"Average Return (w={window})")
    axes[0].set_title("Learning Curve - Vessel Tracking DQN")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps_train, smooth(q_history, window), color="darkorange", linewidth=1.5)
    axes[1].set_ylabel(f"Average Q (w={window})")
    axes[1].set_xlabel("Episode")
    axes[1].grid(True, alpha=0.3)

    if val_history:
        val_eps  = [v["ep"]   for v in val_history]
        val_dice = [v["dice"] for v in val_history]
        val_prec = [v["precision"] for v in val_history]
        val_rec  = [v["recall"]    for v in val_history]
        axes[2].plot(val_eps, val_dice, label="Dice",      color="green",  linewidth=1.5)
        axes[2].plot(val_eps, val_prec, label="Precision", color="purple", linewidth=1.2, linestyle="--")
        axes[2].plot(val_eps, val_rec,  label="Recall",    color="teal",   linewidth=1.2, linestyle=":")
        axes[2].set_ylabel("Val Metric")
        axes[2].set_xlabel("Episode")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "learning_curve.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_validation(val_samples, agent):
    """val set 전체에 대해 MultiAgentTracker로 Dice/Prec/Recall 계산."""
    tracker = MultiAgentTracker(agent)
    prev_eps = agent.epsilon
    agent.epsilon = 0.0

    metrics_list = []
    for i in range(len(val_samples)):
        image, gray_padded, mask, skeleton, _, branch_map, near_skel, endpoint_map, distance_map = val_samples[i]
        visited, _, _ = tracker.track(
            image, mask, skeleton,
            branch_map=branch_map, near_skel=near_skel, endpoint_map=endpoint_map,
            distance_map=distance_map,
        )
        m = compute_overlap_metrics(visited, skeleton)
        metrics_list.append(m)

    agent.epsilon = prev_eps
    avg = {k: float(np.mean([m[k] for m in metrics_list]))
           for k in ["iou", "dice", "precision", "recall"]}
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true", help="실시간 시각화 창 표시")
    parser.add_argument("--reset_epsilon", action="store_true", help="epsilon을 초기값으로 리셋")
    args = parser.parse_args()

    set_seed(CONFIG["random_seed"])
    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG["device"] = device
    print(f"디바이스: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # 고정 입력 크기(21x21)에 최적 커널 캐싱

    if args.render:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 565, 584)
        print("시각화 창 활성화 - 창을 닫으면 학습도 종료됩니다.")

    # 데이터셋 로드 (train/val 분할)
    train_samples, val_samples = build_registry(CONFIG, seed=CONFIG["random_seed"])
    agent = DQNAgent(CONFIG)

    start_episode = 0
    if args.resume:
        start_episode = agent.load(args.resume)
        print(f"체크포인트 로드: {args.resume} (ep {start_episode})")
        if args.reset_epsilon:
            agent.epsilon = CONFIG["epsilon_start"]
            agent.step_count = 0
            print(f"epsilon 리셋: {agent.epsilon}")

    total_steps_ref = [0]
    reward_history  = []
    q_history       = []
    steps_history   = []
    val_history     = []   # {"ep", "dice", "precision", "recall", "iou"}

    ep_bar = tqdm(
        range(start_episode + 1, CONFIG["n_episodes"] + 1),
        desc="Training",
        unit="ep",
        dynamic_ncols=True,
    )

    for ep in ep_bar:
        # train set에서 랜덤 샘플 (on-demand 로드)
        sample = train_samples[random.randrange(len(train_samples))]

        ep_reward, avg_loss, avg_q, ep_steps, ep_spawned, _, _, window_closed = run_ma_episode(
            agent, sample, CONFIG, total_steps_ref, ep=ep, do_render=args.render
        )

        if window_closed:
            print("창 닫힘 - 학습 종료")
            break

        reward_history.append(ep_reward)
        q_history.append(avg_q)
        steps_history.append(ep_steps)
        avg_reward = np.mean(reward_history[-50:])
        avg_steps  = np.mean(steps_history[-50:])

        ep_bar.set_postfix({
            "reward":  f"{ep_reward:.1f}",
            "avg50":   f"{avg_reward:.1f}",
            "loss":    f"{avg_loss:.4f}",
            "e":       f"{agent.epsilon:.3f}",
            "spawned": ep_spawned,
            "steps":   f"{avg_steps:.0f}",
        })

        if ep % CONFIG["log_interval"] == 0:
            tqdm.write(
                f"[ep {ep:5d}] avg_reward {avg_reward:7.1f} | loss {avg_loss:.4f} "
                f"| e {agent.epsilon:.4f} | steps {avg_steps:.0f}"
            )

        # Validation
        if ep % CONFIG["val_interval"] == 0 and len(val_samples) > 0:
            val_m = run_validation(val_samples, agent)
            val_history.append({"ep": ep, **val_m})
            tqdm.write(
                f"  [VAL ep {ep:5d}] Dice {val_m['dice']:.4f} | "
                f"Prec {val_m['precision']:.4f} | Recall {val_m['recall']:.4f} | "
                f"IoU {val_m['iou']:.4f}"
            )

        if ep % CONFIG["save_interval"] == 0:
            ckpt_path = os.path.join(CONFIG["checkpoints_dir"], f"ep_{ep:05d}.pt")
            agent.save(ckpt_path, ep)
            tqdm.write(f"  → 체크포인트 저장: {ckpt_path}")
            curve_path = save_learning_curve(
                reward_history, q_history, val_history, CONFIG["results_dir"]
            )
            tqdm.write(f"  → 학습 곡선 저장: {curve_path}")

    if args.render:
        cv2.destroyAllWindows()
    print("학습 완료.")


if __name__ == "__main__":
    main()
