"""
Optic Disc Segmentation — Lightweight U-Net (PyTorch).
입력: (B, 3, 256, 256) float32, 정규화된 RGB
출력: (B, 1, 256, 256) float32, sigmoid (disc 확률맵)
"""
import numpy as np
import cv2
import torch
import torch.nn as nn


_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
INPUT_SIZE = 256


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DiscSegNet(nn.Module):
    """3-level U-Net: ~3M 파라미터, 빠른 추론."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = _ConvBlock(3,   32)
        self.enc2 = _ConvBlock(32,  64)
        self.enc3 = _ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _ConvBlock(128, 256)

        # Decoder
        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = _ConvBlock(256, 128)
        self.up2   = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.dec2  = _ConvBlock(128, 64)
        self.up1   = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)
        self.dec1  = _ConvBlock(64,  32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# ── 추론 헬퍼 ─────────────────────────────────────────────────────────
def predict_disc_center(model, image_np, mask_np, device):
    """
    Parameters
    ----------
    image_np : (H, W, 3) float32 [0,1]  — RGB
    mask_np  : (H, W) bool               — FOV 마스크
    device   : torch.device

    Returns
    -------
    (r, c) : int  — 원본 이미지 좌표계의 disc 중심
    None          — 예측 실패 시 (HoughCircles fallback용)
    """
    H, W = image_np.shape[:2]

    # 256×256으로 리사이즈
    img_resized = cv2.resize(
        (image_np * 255).astype(np.uint8),
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0

    # 정규화
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1)  # (3,256,256)
    img_t = (img_t - _MEAN[:, None, None]) / _STD[:, None, None]
    img_t = img_t.unsqueeze(0).to(device)                   # (1,3,256,256)

    with torch.no_grad():
        pred = model(img_t)[0, 0].cpu().numpy()             # (256,256)

    disc_mask = pred > 0.5

    if not disc_mask.any():
        return None

    # centroid 계산
    ys, xs = np.where(disc_mask)
    cy_256 = float(ys.mean())
    cx_256 = float(xs.mean())

    # 원본 좌표로 스케일 백
    r = int(round(cy_256 * H / INPUT_SIZE))
    c = int(round(cx_256 * W / INPUT_SIZE))

    # FOV 안으로 클램핑
    r = max(0, min(H - 1, r))
    c = max(0, min(W - 1, c))

    if not mask_np[r, c]:
        return None

    return (r, c)
