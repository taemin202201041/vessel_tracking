"""
멀티 에이전트 혈관 추적기.
- optic disc에서 시작
- 분기점 도달 시 새 에이전트 스폰
- 공유 방문맵으로 중복 탐색 방지
- 배치 추론으로 GPU 효율화
"""
import numpy as np
from collections import deque

from src.vessel_utils import (
    DIRECTIONS,
    find_branch_points,
    find_start_in_optic_disc,
    get_skeleton_directions,
)
from config import CONFIG

MAX_REVISIT = 10


def trim_tail(trajectory, skeleton):
    """
    trajectory에서 마지막 skeleton 픽셀 이후의 꼬리를 분리한다.

    Returns
    -------
    vessel_part : list[(r,c)]  — skeleton 위를 밟은 마지막 지점까지
    tail_part   : list[(r,c)]  — 그 이후 이탈 구간 (빈 리스트일 수 있음)
    """
    last_on = -1
    for i, (r, c) in enumerate(trajectory):
        if skeleton[r, c]:
            last_on = i
    if last_on == -1:
        return [], list(trajectory)          # 전체가 꼬리
    return list(trajectory[:last_on + 1]), list(trajectory[last_on + 1:])


SPAWN_GRACE = 10  # 스폰된 에이전트에게 주는 off_vessel 유예 스텝 수

class _Agent:
    def __init__(self, pos, last_dir=0, is_spawned=False, local_window=20):
        self.pos = pos
        self.last_dir = last_dir
        self.steps = 0
        self.trajectory = [pos]
        self.revisit_streak = 0
        self.off_vessel_streak = -SPAWN_GRACE if is_spawned else 0
        self._recent_deque = deque(maxlen=local_window)


class MultiAgentTracker:
    def __init__(self, dqn_agent):
        cfg = CONFIG
        self.agent = dqn_agent
        self.patch_size = cfg["patch_size"]
        self.half = self.patch_size // 2
        self.max_steps = cfg["max_steps"]
        self.max_agents = cfg.get("max_agents", 100)
        self.max_off_vessel = cfg.get("max_off_vessel", 20)
        n_actions = cfg["n_actions"]
        p = self.patch_size
        # 방향별 dir_map 미리 계산
        self._dir_maps = [np.full((p, p), d / (n_actions - 1), dtype=np.float32)
                          for d in range(n_actions)]
        # obs 버퍼 미리 할당
        self._obs_buf = np.empty((self.max_agents, 3, p, p), dtype=np.float32)

    # ------------------------------------------------------------------
    def _get_obs(self, gray_padded, visited_padded, pos, last_dir, out):
        """gray_padded/visited_padded 슬라이싱으로 경계 체크 없이 고속 관찰 생성."""
        r, c = pos
        p = self.patch_size
        out[0] = gray_padded[r:r + p, c:c + p]
        out[1] = self._dir_maps[last_dir]
        out[2] = visited_padded[r:r + p, c:c + p]

    # ------------------------------------------------------------------
    def track(self, image, mask, skeleton=None, branch_map=None, near_skel=None, endpoint_map=None):
        """
        Parameters
        ----------
        image    : (H, W, 3) float32
        mask     : (H, W) bool  — FOV 마스크
        skeleton : (H, W) bool or None

        Returns
        -------
        visited       : (H, W) bool  — 방문된 픽셀
        trajectories  : list of list of (r, c) tuples
        """
        H, W = image.shape[:2]
        h = self.half
        p = self.patch_size
        gray = image[:, :, 1]  # green 채널 — 혈관이 어둡게 선명히 보임
        gray_padded = np.pad(gray, h, mode='constant', constant_values=0.0)
        visited = np.zeros((H, W), dtype=bool)
        visited_padded = np.zeros((H + 2 * h, W + 2 * h), dtype=bool)

        # 시작점: disc 안 어두운 픽셀 기반 (skeleton 있으면 GT로 검증, 없으면 이미지만으로)
        start = find_start_in_optic_disc(image, mask, skeleton=skeleton)
        # branch_map: 전달된 값 사용, 없으면 직접 계산
        if branch_map is None and skeleton is not None:
            branch_map = find_branch_points(skeleton)

        # 시작점이 FOV 밖이면 FOV 안 랜덤으로
        if not mask[start]:
            pts = np.argwhere(mask)
            start = tuple(pts[np.random.randint(len(pts))])

        sr, sc = start
        visited[sr, sc] = True
        visited_padded[sr + h, sc + h] = True
        # 시작점을 분기점처럼 취급: 스켈레톤이 뻗는 모든 방향으로 에이전트 뿌리기
        if skeleton is not None:
            start_dirs = get_skeleton_directions(skeleton, sr, sc)
        else:
            start_dirs = []
        local_window = CONFIG.get("local_revisit_window", 20)
        processed_branches = set()
        active = []
        for d in (start_dirs if start_dirs else [0]):
            sdr, sdc = DIRECTIONS[d]
            snr, snc = sr + sdr, sc + sdc
            if 0 <= snr < H and 0 <= snc < W and mask[snr, snc] and not visited[snr, snc]:
                ag = _Agent((snr, snc), last_dir=d, local_window=local_window)
                ag.trajectory = [(sr, sc), (snr, snc)]
                visited[snr, snc] = True
                visited_padded[snr + h, snc + h] = True
                active.append(ag)
        if not active:
            initial_dir = start_dirs[0] if start_dirs else 0
            active = [_Agent(start, last_dir=initial_dir, local_window=local_window)]
        all_trajectories = []
        total_spawned = 0

        # ------------------------------------------------------------------
        while active:
            n = len(active)
            for i, ag in enumerate(active):
                self._get_obs(gray_padded, visited_padded, ag.pos, ag.last_dir, self._obs_buf[i])
            actions = self.agent.select_actions_batch(self._obs_buf[:n])

            next_active = []
            spawned = []

            for ag, action in zip(active, actions):
                r, c = ag.pos
                dr, dc = DIRECTIONS[action]
                nr, nc = r + dr, c + dc
                ag.steps += 1
                ag.last_dir = action

                # 경계 / FOV 이탈 → 종료
                if not (0 <= nr < H and 0 <= nc < W) or not mask[nr, nc]:
                    all_trajectories.append(ag.trajectory)
                    continue

                # 재방문
                if visited[nr, nc]:
                    ag.revisit_streak += 1
                    ag.pos = (nr, nc)
                    if ag.revisit_streak >= MAX_REVISIT or ag.steps >= self.max_steps:
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
                ag._recent_deque.append((nr, nc))

                # off_vessel_streak 갱신
                if skeleton is not None:
                    if skeleton[nr, nc] or (near_skel is not None and near_skel[nr, nc]):
                        ag.off_vessel_streak = 0
                    else:
                        ag.off_vessel_streak += 1

                # 혈관 끝점 도달 → 즉시 종료
                if endpoint_map is not None and endpoint_map[nr, nc]:
                    all_trajectories.append(ag.trajectory)
                    continue

                # 혈관 완전 이탈 연속 N스텝 → 종료
                if ag.off_vessel_streak >= self.max_off_vessel:
                    all_trajectories.append(ag.trajectory)
                    continue

                # 분기점: 현재 위치 또는 인접 스켈레톤 픽셀이 분기점이면 스폰
                branch_pos = None
                if skeleton is not None and branch_map is not None:
                    if branch_map[nr, nc]:
                        branch_pos = (nr, nc)
                    else:
                        r0b, r1b = max(0, nr-3), min(H, nr+4)
                        c0b, c1b = max(0, nc-3), min(W, nc+4)
                        sub = branch_map[r0b:r1b, c0b:c1b]
                        if sub.any():
                            pts = np.argwhere(sub)
                            dists = (pts[:,0]-(nr-r0b))**2 + (pts[:,1]-(nc-c0b))**2
                            best = pts[np.argmin(dists)]
                            branch_pos = (r0b+int(best[0]), c0b+int(best[1]))

                if branch_pos is not None and branch_pos not in processed_branches:
                    processed_branches.add(branch_pos)
                    br, bc = branch_pos
                    if (nr, nc) == (br, bc):
                        came_from = (action + 4) % 8
                    else:
                        dr_b, dc_b = nr - br, nc - bc
                        came_from = min(range(len(DIRECTIONS)),
                                        key=lambda d: (DIRECTIONS[d][0]-dr_b)**2 + (DIRECTIONS[d][1]-dc_b)**2)
                    branch_dirs = get_skeleton_directions(skeleton, br, bc, came_from_dir=came_from)
                    n_spawned_here = 0
                    for bd in branch_dirs:
                        if len(next_active) + len(spawned) >= self.max_agents:
                            break
                        bdr, bdc = DIRECTIONS[bd]
                        bnr, bnc = br + bdr, bc + bdc
                        if 0 <= bnr < H and 0 <= bnc < W and mask[bnr, bnc] and not visited[bnr, bnc]:
                            new_ag = _Agent((bnr, bnc), last_dir=bd, is_spawned=True,
                                            local_window=local_window)
                            new_ag.trajectory = [(br, bc), (bnr, bnc)]
                            visited[bnr, bnc] = True
                            visited_padded[bnr + h, bnc + h] = True
                            spawned.append(new_ag)
                            total_spawned += 1
                            n_spawned_here += 1
                    # 실제로 스폰된 경우에만 원래 에이전트 종료
                    if n_spawned_here > 0:
                        all_trajectories.append(ag.trajectory)
                        continue

                # max_steps 도달
                if ag.steps >= self.max_steps:
                    all_trajectories.append(ag.trajectory)
                else:
                    next_active.append(ag)

            active = next_active + spawned

        # ── 꼬리 분리 (시각화용) ─────────────────────────────────────
        # skeleton이 없으면 트리밍 불가 → 원본 그대로 반환
        if skeleton is None:
            return visited, all_trajectories, []

        trimmed, tails = [], []
        for traj in all_trajectories:
            vessel, tail = trim_tail(traj, skeleton)
            trimmed.append(vessel)
            tails.append(tail)

        # 메트릭은 raw visited 기준 (trimmed는 시각화 색칠용만)
        return visited, trimmed, tails
