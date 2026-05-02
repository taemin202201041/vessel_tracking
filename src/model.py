"""
CNN 기반 Dueling DQN — SE + Residual + Dilated Conv 백본.
입력: (batch, 3, PATCH_SIZE, PATCH_SIZE)
출력: Q(s, a) = V(s) + A(s,a) - mean(A(s,·))
"""
import torch
import torch.nn as nn


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation: 채널 어텐션."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class _ResidualDilatedBlock(nn.Module):
    """Residual + Dilated Conv3×3 + SE."""
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = _SEBlock(out_ch)
        self.proj = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.proj(x))


class DuelingDQN(nn.Module):
    def __init__(self, patch_size: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        self.backbone = nn.Sequential(
            # Stem: 31×31 → 15×15 (먼저 다운샘플해서 연산량 절감)
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # → 15×15
            # Dilated blocks (15×15에서 적용 — 31×31보다 4배 빠름)
            _ResidualDilatedBlock(32, 64,  dilation=1),   # 일반
            _ResidualDilatedBlock(64, 64,  dilation=2),   # receptive field 확장
            nn.MaxPool2d(2),                              # → 7×7
            _ResidualDilatedBlock(64, 128, dilation=1),   # 채널 확장 + 정제
        )

        dummy = torch.zeros(1, 3, patch_size, patch_size)
        flat_size = self.backbone(dummy).view(1, -1).shape[1]

        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        value     = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
