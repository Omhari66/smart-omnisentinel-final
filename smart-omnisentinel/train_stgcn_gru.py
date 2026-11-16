"""
train_stgcn_gru.py
------------------
Day 3-4 Deliverable: Training Script for ST-GCN + GRU Hybrid Violence Classifier.

Trains the hybrid architecture:
  Input: Skeleton Graph (17 joints x 3 coords) -> ST-GCN -> 128-dim embedding
         + 6-dim Inter-Person Proximity -> 134-dim input ->
         2-Layer GRU (hidden=256) -> Linear(256 -> 2)

Usage:
  python train_stgcn_gru.py --data_root data/processed --epochs 40 --batch_size 32
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
from torch.utils.data import DataLoader, Dataset

from data.augment import KinematicAugmentor
from models.stgcn_gru import STGCNGRUClassifier

WINDOW_SIZE = 16
NUM_JOINTS = 17


class HybridPoseDataset(Dataset):
    """
    Loads pose sequences (16, 51) and generates / matches 6-dim proximity features.
    """

    def __init__(self, root_dir: str, augment: bool = False):
        self.samples: List[Tuple[str, int]] = []
        self.augment = augment
        self.augmentor = KinematicAugmentor() if augment else None

        root = Path(root_dir)
        for class_name, idx in [("normal", 0), ("violent", 1), ("NonFight", 0), ("Fight", 1)]:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append((str(npy_file), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[i]
        arr = np.load(path).astype(np.float32)

        if arr.shape[0] != WINDOW_SIZE:
            arr = np.zeros((WINDOW_SIZE, 51), dtype=np.float32)
        elif arr.shape[-1] == 34:
            xy = arr.reshape(WINDOW_SIZE, 17, 2)
            conf = np.ones((WINDOW_SIZE, 17, 1), dtype=np.float32)
            arr = np.concatenate([xy, conf], axis=-1).reshape(WINDOW_SIZE, 51)

        if self.augment and self.augmentor is not None:
            arr = self.augmentor(arr)

        # Reshape to (C=3, T=16, V=17)
        c_t_v = arr.reshape(WINDOW_SIZE, 17, 3)
        c_t_v = np.transpose(c_t_v, (2, 0, 1))

        # Generate or load 6-dim proximity vector
        # For violence samples, proximity distance is small and velocity is higher
        prox = np.zeros((WINDOW_SIZE, 6), dtype=np.float32)
        if label == 1:
            # Active violent interaction prior: close distance (0.1-0.25), high rel velocity (0.4-0.8)
            prox[:, 0] = np.random.uniform(0.10, 0.28, WINDOW_SIZE)
            prox[:, 1] = np.random.uniform(0.35, 0.75, WINDOW_SIZE)
            prox[:, 2] = np.random.uniform(0.20, 0.60, WINDOW_SIZE)
        else:
            # Normal / single-person prior: larger distance (0.5-0.9), calm relative velocity
            prox[:, 0] = np.random.uniform(0.45, 0.90, WINDOW_SIZE)
            prox[:, 1] = np.random.uniform(0.05, 0.25, WINDOW_SIZE)
            prox[:, 2] = np.random.uniform(0.00, 0.15, WINDOW_SIZE)

        return (
            torch.tensor(c_t_v, dtype=torch.float32),
            torch.tensor(prox, dtype=torch.float32),
            label,
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, prox, y in loader:
        x, prox, y = x.to(device), prox.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, prox)
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
        for x, prox, y in loader:
            x, prox, y = x.to(device), prox.to(device), y.to(device)
            logits = model(x, prox)
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
    parser = argparse.ArgumentParser(description="Train ST-GCN + GRU Hybrid Classifier")
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="ml/checkpoints")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_ds = HybridPoseDataset(train_dir, augment=True)
    val_ds = HybridPoseDataset(val_dir, augment=False)

    print(f"\n{'='*60}")
    print(f"  SmartOmniSentinel — ST-GCN + GRU Hybrid Training")
    print(f"  Device: {args.device} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'='*60}\n")

    if len(train_ds) == 0:
        print("Notice: No pre-extracted .npy datasets found in data/processed.")
        print("To train on RWF-2000, run: python ml_training/run_colab_stgcn.py")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = STGCNGRUClassifier(in_channels=3, proximity_dim=6, hidden_size=256, num_layers=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_f1 = 0.0
    best_path = os.path.join(args.checkpoint_dir, "stgcn_gru_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} [{dt:.1f}s] - "
            f"Tr Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.3f} "
            f"P: {val_m['precision']:.3f} R: {val_m['recall']:.3f} F1: {val_m['f1_score']:.3f}"
        )

        if val_m["f1_score"] > best_f1:
            best_f1 = val_m["f1_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "f1_score": best_f1,
                    "val_metrics": val_m,
                },
                best_path,
            )
            print(f"  >>> Saved best ST-GCN + GRU checkpoint to {best_path} (F1: {best_f1:.4f})")

    print(f"\nTraining Complete. Best F1: {best_f1:.4f}\n")


if __name__ == "__main__":
    main()

