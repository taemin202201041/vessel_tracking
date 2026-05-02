"""
DQN 에이전트: ReplayBuffer, epsilon-greedy, Double DQN 학습, 체크포인트 저장/로드.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast

from src.model import DuelingDQN


class ReplayBuffer:
    """프리얼로케이션 링버퍼 — deque+zip+np.array 오버헤드 제거."""

    def __init__(self, capacity: int, obs_shape: tuple = (3, 21, 21)):
        self.capacity = capacity
        self.ptr  = 0
        self.size = 0
        self.states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

        patch_size = config["patch_size"]
        n_actions = config["n_actions"]

        self.online_net = DuelingDQN(patch_size, n_actions).to(self.device)
        self.target_net = DuelingDQN(patch_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # torch.compile: Windows에서는 Triton 미지원으로 비활성화
        # Linux 환경이면 아래 주석 해제 시 10-20% 속도향상 가능
        # self.online_net = torch.compile(self.online_net)
        # self.target_net = torch.compile(self.target_net)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config["learning_rate"])
        p = config["patch_size"]
        self.buffer = ReplayBuffer(config["replay_buffer"], obs_shape=(3, p, p))

        # AMP: float16 forward pass로 메모리 대역폭 절감
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.epsilon = config["epsilon_start"]
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.cfg["n_actions"])
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(state_t)
        return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.cfg["batch_size"]:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg["batch_size"])

        states_t      = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions_t     = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards_t     = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        next_states_t = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones_t       = torch.from_numpy(dones).to(self.device, non_blocking=True)

        # AMP: float16 forward pass
        with autocast('cuda', enabled=self.use_amp):
            q_values = self.online_net(states_t)
            q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = self.online_net(next_states_t).argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = rewards_t + self.cfg["gamma"] * next_q * (1.0 - dones_t)

            loss = F.smooth_l1_loss(q_sa, target)
            avg_q = q_values.mean().item()

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
            self.optimizer.step()

        self.step_count += 1

        # Target network 업데이트
        if self.step_count % self.cfg["target_update"] == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Epsilon 감소
        self.epsilon = max(
            self.cfg["epsilon_end"],
            self.epsilon * self.cfg["epsilon_decay"],
        )

        return loss.item(), avg_q

    def save(self, path: str, episode: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "episode":    episode,
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "step_count": self.step_count,
        }, path)

    def select_actions_batch(self, states: np.ndarray) -> list:
        """N개 observation을 한 번에 배치 추론."""
        states_t = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(states_t)
        return q_values.argmax(dim=1).cpu().numpy().tolist()

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.step_count = ckpt["step_count"]
        return ckpt["episode"]
