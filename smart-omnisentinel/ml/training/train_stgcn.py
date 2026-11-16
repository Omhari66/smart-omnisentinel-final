"""
ml/training/train_stgcn.py
--------------------------
Training script for Spatio-Temporal Graph Convolutional Network (ST-GCN)
on RWF-2000 / RLVS skeleton sequences.

Usage:
    python -m ml.training.train_stgcn \
        --data_root data/processed \
        --epochs 40 \
        --batch_size 32 \
        --lr 1e-3 \
        --device cuda \
        --checkpoint_dir ml/checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ml.models.st_gcn import NUM_COCO_JOINTS, STGCNClassifier

CLASS_NAMES = ["normal", "violent"]
CLASS_TO_IDX = {"normal": 0, "violent": 1, "NonFight": 0, "Fight": 1}
WINDOW_SIZE = 16


class STGCNDataset(Dataset):
    """
    Loads pre-extracted (16, 51) or (16, 34) keypoint windows.
    Automatically reshapes to (in_channels=3, time=16, joints=17).
    """

    def __init__(self, root_dir: str, augment: bool = False):
        self.samples: List[Tuple[str, int]] = []
        self.augment = augment

        root = Path(root_dir)
        for class_name, idx in [("normal", 0), ("violent", 1), ("NonFight", 0), ("Fight", 1)]:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append((str(npy_file), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        arr = np.load(path).astype(np.float32)

        # Standardize to (16, 51)
        if arr.shape[0] != WINDOW_SIZE:
            arr = np.zeros((WINDOW_SIZE, 51), dtype=np.float32)
        elif arr.shape[-1] == 34:
            # Append confidence 1.0
            xy = arr.reshape(WINDOW_SIZE, 17, 2)
            conf = np.ones((WINDOW_SIZE, 17, 1), dtype=np.float32)
            arr = np.concatenate([xy, conf], axis=-1).reshape(WINDOW_SIZE, 51)

        if self.augment:
            arr = self._augment(arr)

        # Reshape to (C=3, T=16, V=17)
        c_t_v = arr.reshape(WINDOW_SIZE, 17, 3)
        c_t_v = np.transpose(c_t_v, (2, 0, 1))  # (3, 16, 17)

        return torch.tensor(c_t_v, dtype=torch.float32), label

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy().reshape(WINDOW_SIZE, 17, 3)

        # 1. Coordinate jitter on X, Y
        noise = np.random.normal(0, 0.015, (WINDOW_SIZE, 17, 2))
        arr[:, :, :2] += noise
        arr[:, :, :2] = np.clip(arr[:, :, :2], 0.0, 1.0)

        # 2. Random horizontal flip (mirror X coords + swap symmetrical joints)
        if np.random.rand() < 0.5:
            arr[:, :, 0] = 1.0 - arr[:, :, 0]
            # Swap left/right pairs: (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
            lr_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for left, right in lr_pairs:
                tmp = arr[:, left, :].copy()
                arr[:, left, :] = arr[:, right, :]
                arr[:, right, :] = tmp

        # 3. Temporal speed jitter
        if np.random.rand() < 0.3:
            drop_idx = np.random.randint(1, WINDOW_SIZE - 1)
            arr = np.delete(arr, drop_idx, axis=0)
            arr = np.insert(arr, drop_idx, arr[drop_idx - 1], axis=0)

        return arr.reshape(WINDOW_SIZE, 51)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / max(total, 1), correct / max(total, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)

            for p, t in zip(preds.cpu().numpy(), y.cpu().numpy()):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 1:
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN Violence Detection Classifier")
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="ml/checkpoints")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_ds = STGCNDataset(train_dir, augment=True)
    val_ds = STGCNDataset(val_dir, augment=False)

    print(f"\n--- ST-GCN Model Training ---")
    print(f"Device: {args.device} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training samples found. Please extract keypoints first.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = STGCNClassifier(in_channels=3, num_classes=2, proximity_dim=0).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_f1 = 0.0
    best_ckpt = os.path.join(args.checkpoint_dir, "stgcn_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} [{elapsed:.1f}s] - "
            f"Tr Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} "
            f"P: {val_metrics['precision']:.3f} R: {val_metrics['recall']:.3f} F1: {val_metrics['f1_score']:.3f}"
        )

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "f1_score": best_f1,
                    "val_metrics": val_metrics,
                },
                best_ckpt,
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt} (F1: {best_f1:.4f})")

    print(f"\nTraining complete! Best F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
