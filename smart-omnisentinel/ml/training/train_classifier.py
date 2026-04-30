"""
ml/training/train_classifier.py
---------------------------------
Training script for the TemporalClassifier (CNN+GRU).

Usage:
    python -m ml.training.train_classifier \
        --data_root data/processed \
        --epochs 50 \
        --batch_size 32 \
        --lr 1e-3 \
        --device cuda \
        --checkpoint_dir ml/checkpoints

Expected data layout:
    data/processed/
        train/
            normal/          # .npy files, shape (16, 55)
            violent/
            distress/
        val/
            normal/
            violent/
            distress/

Each .npy file is one 16-frame sliding window of features.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ml.models.temporal_model import TemporalClassifier, WINDOW_SIZE, FEATURE_DIM

CLASS_NAMES = ["normal", "violent"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FeatureWindowDataset(Dataset):
    """
    Loads pre-extracted (WINDOW_SIZE, FEATURE_DIM) numpy arrays.
    Each file is one training example.
    """

    def __init__(self, root_dir: str, augment: bool = False):
        self.samples: List[Tuple[str, int]] = []
        self.augment = augment

        for class_name in CLASS_NAMES:
            class_dir = Path(root_dir) / class_name
            if not class_dir.exists():
                continue
            idx = CLASS_TO_IDX[class_name]
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append((str(npy_file), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        arr = np.load(path).astype(np.float32)

        # Validate shape
        if arr.shape != (WINDOW_SIZE, FEATURE_DIM):
            # Pad or truncate if needed
            arr = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)

        if self.augment:
            arr = self._augment(arr)

        return torch.tensor(arr), label

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """
        Data augmentation for temporal sequences.
        - Gaussian noise on keypoint coordinates
        - Temporal jitter (slight speed perturbation)
        - Random horizontal flip (mirror all X coordinates)
        """
        arr = arr.copy()

        # Gaussian noise (applied to keypoint x,y columns only)
        noise = np.random.normal(0, 0.01, arr[:, :34].shape)
        arr[:, :34] += noise
        arr[:, :34] = np.clip(arr[:, :34], 0.0, 1.0)

        # Temporal jitter: randomly drop one frame and repeat a neighbour
        if np.random.rand() < 0.3:
            drop_idx = np.random.randint(1, WINDOW_SIZE - 1)
            arr = np.delete(arr, drop_idx, axis=0)
            arr = np.insert(arr, drop_idx, arr[drop_idx - 1], axis=0)

        # Horizontal flip: mirror X coordinates (every other value in [0:34])
        if np.random.rand() < 0.5:
            arr[:, 0:34:2] = 1.0 - arr[:, 0:34:2]

        return arr

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for WeightedRandomSampler."""
        counts = [0] * len(CLASS_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts)
        weights = [total / (c + 1e-6) for c in counts]
        return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: TemporalClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


def evaluate(
    model: TemporalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Per-class metrics
    from collections import Counter
    metrics = {"loss": total_loss / total, "acc": correct / total}

    # Simple precision/recall per class
    for c, name in enumerate(CLASS_NAMES):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != c and l == c)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = f1

    return metrics


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Datasets
    train_ds = FeatureWindowDataset(
        os.path.join(args.data_root, "train"), augment=True
    )
    val_ds = FeatureWindowDataset(
        os.path.join(args.data_root, "val"), augment=False
    )

    if len(train_ds) == 0:
        print("ERROR: No training samples found. Check data_root path.")
        return

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Weighted sampler for class imbalance
    class_weights = train_ds.get_class_weights()
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Model
    model = TemporalClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss: weighted cross-entropy for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    # Training
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_f1 = 0.0
    best_checkpoint = os.path.join(args.checkpoint_dir, "temporal_v1.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Use mean of per-class F1 as primary metric
        val_f1 = np.mean([
            val_metrics.get(f"{c}_f1", 0.0) for c in CLASS_NAMES
        ])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} "
            f"mean_f1={val_f1:.3f} | {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "best_val_f1": best_val_f1,
                },
                best_checkpoint,
            )
            print(f"  ↑ Best checkpoint saved (mean F1={val_f1:.4f})")

    print(f"\nTraining complete. Best val mean_F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartOmniSentinel Temporal Classifier")
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="ml/checkpoints")
    args = parser.parse_args()
    main(args)
