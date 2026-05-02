"""
CNN 기반 Dueling DQN 모델 아키텍처.
입력: (batch, 3, PATCH_SIZE, PATCH_SIZE)
출력: Q(s, a) = V(s) + A(s,a) - mean(A(s,·))
"""
import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, patch_size: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # → (32, P, P)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64, P, P)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (64, P//2, P//2)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64, P//2, P//2)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (64, P//4, P//4)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64, P//4, P//4)
            nn.ReLU(),
        )

        # Flatten 크기 계산
        dummy = torch.zeros(1, 3, patch_size, patch_size)
        cnn_out = self.cnn(dummy)
        flat_size = cnn_out.view(1, -1).shape[1]

        # Value Stream: FC → 1
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage Stream: FC → N_ACTIONS
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        feat = self.cnn(x).flatten(1)
        value = self.value_stream(feat)                      # (B, 1)
        advantage = self.advantage_stream(feat)              # (B, A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
