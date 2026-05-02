"""
Gymnasium 기반 망막 혈관 추적 환경.
관찰: (3, PATCH_SIZE, PATCH_SIZE) - [로컬패치, 방향히스토리, 방문맵]
행동: Discrete(8) - 8방향 이동
"""
import os
import random

import numpy as np
import cv2
from PIL import Image as PILImage
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from src.vessel_utils import find_start_in_optic_disc, find_branch_points, find_endpoints


def _correct_illumination(img, mask):
    """조명 불균일 보정: v'_p = v_p * (v_m / v^b_p)
    배경(v^b_p)은 강한 가우시안 블러로 추정, v_m은 FOV 내 채널 평균."""
    fov = mask.astype(np.float32)
    corrected = np.empty_like(img)
    for ch in range(3):
        channel = img[:, :, ch]
        # 배경 추정: 커널 크기는 홀수여야 함
        bg = cv2.GaussianBlur(channel, (101, 101), 30)
        bg = np.where(bg < 1e-6, 1e-6, bg)  # 0 나누기 방지
        # FOV 내 평균
        mean_val = float(channel[mask].mean()) if mask.any() else 1.0
        corrected[:, :, ch] = np.clip(channel * (mean_val / bg) * fov, 0, 1)
    return corrected

# 8방향 이동 벡터 (행, 열) — 0:상, 1:우상, ..., 7:좌상
DIRECTIONS = [
    (-1,  0), (-1,  1), (0,  1), (1,  1),
    ( 1,  0), ( 1, -1), (0, -1), (-1, -1),
]

MAX_REVISIT = 10


class RetinalVesselEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, split: str = "train"):
        super().__init__()
        assert split in ("train", "test")
        self.split = split
        cfg = CONFIG

        self.patch_size = cfg["patch_size"]
        self.half = self.patch_size // 2
        self.n_actions = cfg["n_actions"]
        self.max_steps = cfg["max_steps"]

        self.r_on        = cfg["reward_on_vessel"]
        self.r_near      = cfg["reward_near_vessel"]
        self.r_off       = cfg["reward_off_vessel"]
        self.r_revisit   = cfg["reward_revisit"]
        self.r_boundary  = cfg["reward_boundary"]
        self.r_complete  = cfg["reward_completion"]

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, self.patch_size, self.patch_size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._load_data(cfg, split)

        self.image = None
        self.skeleton = None
        self.mask = None
        self.pos = None
        self.visited = None
        self.visited_padded = None
        self.trajectory = []
        self.step_count = 0
        self.last_dir = 0
        self.revisit_streak = 0

    # ------------------------------------------------------------------
    def _load_data(self, cfg, split):
        img_dir  = cfg[f"{split}_images_dir"]
        mask_dir = cfg[f"{split}_masks_dir"]
        skel_dir = cfg[f"{split}_skeletons_dir"]

        img_files  = sorted(f for f in os.listdir(img_dir)  if f.endswith(".tif"))
        mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".gif"))
        skel_files = sorted(f for f in os.listdir(skel_dir) if f.endswith(".png"))

        if not skel_files:
            raise FileNotFoundError(
                f"스켈레톤 파일이 없습니다: {skel_dir}\n"
                "먼저 preprocess.py를 실행하세요."
            )

        h = self.half
        self.samples = []
        for img_f, mask_f, skel_f in zip(img_files, mask_files, skel_files):
            img  = cv2.imread(os.path.join(img_dir, img_f), cv2.IMREAD_COLOR)
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mask = np.array(PILImage.open(os.path.join(mask_dir, mask_f)).convert("L")) > 127
            img  = _correct_illumination(img, mask)
            gray = img[:,:,1]  # green 채널 — 혈관이 어둡게 선명히 보임
            skel = np.array(PILImage.open(os.path.join(skel_dir, skel_f)).convert("L")) > 127

            # gray를 미리 패딩 → _get_obs에서 단순 슬라이싱만 하면 됨
            gray_padded = np.pad(gray, h, mode='constant', constant_values=0.0)

            start_pos = find_start_in_optic_disc(img, mask, skeleton=skel)
            branch_map = find_branch_points(skel)
            near_skel = cv2.dilate(skel.astype(np.uint8),
                                   np.ones((3, 3), np.uint8)).astype(bool)
            endpoint_map = find_endpoints(skel)
            self.samples.append((img, gray_padded, mask, skel, start_pos,
                                  branch_map, near_skel, endpoint_map))

    # ------------------------------------------------------------------
    def _get_obs(self):
        r, c = self.pos
        p = self.patch_size

        # 채널 0: 단순 슬라이싱 (경계 체크 불필요 — gray 미리 패딩됨)
        patch = self.gray[r:r+p, c:c+p]

        # 채널 1: 방향 히스토리
        dir_map = np.full((p, p), self.last_dir / (self.n_actions - 1), dtype=np.float32)

        # 채널 2: 방문맵 패딩된 버전에서 슬라이싱
        visit_patch = self.visited_padded[r:r+p, c:c+p].astype(np.float32)

        return np.stack([patch, dir_map, visit_patch], axis=0)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.samples))
        self.image, self.gray, self.mask, self.skeleton, optic_disc, _, self.near_skel, _ = self.samples[idx]
        H, W = self.skeleton.shape
        h = self.half

        self.visited = np.zeros((H, W), dtype=bool)
        self.visited_padded = np.zeros((H + 2*h, W + 2*h), dtype=bool)
        self.trajectory = []
        self.step_count = 0
        self.last_dir = 0
        self.revisit_streak = 0

        # optic disc에서 시작 (평가와 동일한 조건)
        r0, c0 = optic_disc
        # optic disc가 경계에 너무 가까우면 FOV 내 랜덤 스켈레톤 포인트로 대체
        if r0 < h or r0 >= H - h or c0 < h or c0 >= W - h or not self.mask[r0, c0]:
            pts = np.argwhere(self.skeleton)
            valid = pts[(pts[:,0] >= h) & (pts[:,0] < H - h) &
                        (pts[:,1] >= h) & (pts[:,1] < W - h)]
            if len(valid) == 0:
                valid = pts
            chosen = valid[random.randrange(len(valid))]
            r0, c0 = int(chosen[0]), int(chosen[1])
        self.pos = (r0, c0)

        r, c = self.pos
        self.visited[r, c] = True
        self.visited_padded[r + h, c + h] = True
        self.trajectory.append(self.pos)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int):
        r, c = self.pos
        dr, dc = DIRECTIONS[action]
        nr, nc = r + dr, c + dc
        H, W = self.skeleton.shape
        h = self.half
        self.step_count += 1
        self.last_dir = action

        if not (0 <= nr < H and 0 <= nc < W):
            self.revisit_streak = 0
            return self._get_obs(), self.r_boundary, True, False, {"reason": "boundary"}

        if not self.mask[nr, nc]:
            self.revisit_streak = 0
            return self._get_obs(), self.r_boundary, True, False, {"reason": "outside_fov"}

        if self.visited[nr, nc]:
            self.pos = (nr, nc)
            self.revisit_streak += 1
            done = self.revisit_streak >= MAX_REVISIT
            return self._get_obs(), self.r_revisit, done, False, {"reason": "revisit"}

        self.revisit_streak = 0
        self.pos = (nr, nc)
        self.visited[nr, nc] = True
        self.visited_padded[nr + h, nc + h] = True
        self.trajectory.append(self.pos)

        if self.skeleton[nr, nc]:
            reward = self.r_on
        elif self.near_skel[nr, nc]:
            reward = self.r_near
        else:
            reward = self.r_off

        done = self.step_count >= self.max_steps

        if done and len(self.trajectory) > 0:
            on_vessel = sum(1 for pr, pc in self.trajectory if self.skeleton[pr, pc])
            if on_vessel / len(self.trajectory) >= 0.6:
                reward += self.r_complete

        return self._get_obs(), reward, done, False, {}

    # ------------------------------------------------------------------
    def render(self, mode="rgb_array"):
        vis = (self.image * 255).astype(np.uint8).copy()
        for pt in self.trajectory:
            cv2.circle(vis, (pt[1], pt[0]), 1, (255, 255, 255), -1)
        if self.trajectory:
            start = self.trajectory[0]
            cv2.circle(vis, (start[1], start[0]), 4, (0, 255, 0), -1)
            end = self.trajectory[-1]
            cv2.circle(vis, (end[1], end[0]), 4, (0, 0, 255), -1)
        return vis
