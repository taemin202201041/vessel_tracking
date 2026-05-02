"""
Optic Disc Segmentation 학습 스크립트.

실행:
  python train_disc_seg.py
  python train_disc_seg.py --epochs 100 --batch 8

저장: checkpoints/disc_seg.pt  (val Dice 최고 모델)
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from config import CONFIG
from src.disc_model import DiscSegNet, INPUT_SIZE

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── Dataset ──────────────────────────────────────────────────────────
class DiscDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs   = pairs      # [(img_path, mask_path), ...]
        self.augment = augment
        self.normalize = transforms.Normalize(_MEAN, _STD)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.resize(msk, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        msk = (msk > 127).astype(np.float32)

        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        msk_t = torch.from_numpy(msk).unsqueeze(0)

        if self.augment:
            img_t, msk_t = self._augment(img_t, msk_t)

        img_t = self.normalize(img_t)
        return img_t, msk_t

    @staticmethod
    def _augment(img, msk):
        if random.random() > 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
        if random.random() > 0.5:
            img = TF.vflip(img)
            msk = TF.vflip(msk)
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
        msk = TF.rotate(msk, angle)
        # ColorJitter (이미지만)
        brightness = random.uniform(0.8, 1.2)
        contrast   = random.uniform(0.8, 1.2)
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        return img, msk


# ── 손실 함수 ─────────────────────────────────────────────────────────
def dice_loss(pred, target, eps=1e-6):
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    return 1.0 - (2.0 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)


def combined_loss(pred, target):
    bce  = nn.functional.binary_cross_entropy(pred, target, reduction='mean')
    dice = dice_loss(pred, target).mean()
    return 0.5 * bce + 0.5 * dice


# ── 데이터 수집 ───────────────────────────────────────────────────────
def collect_pairs(cfg):
    img_dir  = cfg["full_fundus_dir"]
    mask_dir = cfg["optic_disc_dir"]

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"이미지 폴더 없음: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"마스크 폴더 없음: {mask_dir}")

    mask_files = set(os.listdir(mask_dir))
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if fname in mask_files:
            pairs.append((
                os.path.join(img_dir,  fname),
                os.path.join(mask_dir, fname),
            ))
    print(f"매칭된 쌍: {len(pairs)}개")
    return pairs


# ── 학습 ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    pairs = collect_pairs(CONFIG)
    random.seed(42)
    random.shuffle(pairs)
    n_val  = max(1, int(len(pairs) * 0.2))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"train {len(train_pairs)} / val {len(val_pairs)}")

    train_ds = DiscDataset(train_pairs, augment=True)
    val_ds   = DiscDataset(val_pairs,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = DiscSegNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )

    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)
    save_path = CONFIG.get("disc_seg_checkpoint",
                           os.path.join(CONFIG["checkpoints_dir"], "disc_seg.pt"))

    best_dice = 0.0

    ep_bar = tqdm(range(1, args.epochs + 1), desc="Disc Seg", unit="ep", dynamic_ncols=True)

    for ep in ep_bar:
        # ── Train ──
        model.train()
        train_loss = 0.0
        batch_bar = tqdm(train_dl, desc=f"  train", leave=False, dynamic_ncols=True)
        for imgs, msks in batch_bar:
            imgs, msks = imgs.to(device), msks.to(device)
            preds = model(imgs)
            loss  = combined_loss(preds, msks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_ds)

        # ── Val ──
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, msks in val_dl:
                imgs, msks = imgs.to(device), msks.to(device)
                preds = model(imgs)
                p = (preds > 0.5).float().view(preds.size(0), -1)
                t = msks.view(msks.size(0), -1)
                inter = (p * t).sum(dim=1)
                d = (2.0 * inter) / (p.sum(dim=1) + t.sum(dim=1) + 1e-6)
                val_dice += d.sum().item()
        val_dice /= len(val_ds)

        scheduler.step(val_dice)

        marker = ""
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            marker = " ✓ best"

        ep_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            val_dice=f"{val_dice:.4f}",
            best=f"{best_dice:.4f}",
        )
        tqdm.write(
            f"ep {ep:3d}/{args.epochs} | loss {train_loss:.4f} | "
            f"val Dice {val_dice:.4f} | best {best_dice:.4f}{marker}"
        )

    print(f"\n학습 완료. best val Dice: {best_dice:.4f}")
    print(f"저장 경로: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--lr",     type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
