"""
ml/training/evaluate.py
--------------------------
Standalone evaluation script for a trained TemporalClassifier.
Produces: precision, recall, F1, confusion matrix,
false positive rate, false negative rate per class.

Usage:
    python -m ml.training.evaluate \
        --checkpoint ml/checkpoints/temporal_v1.pt \
        --data_root data/processed/val \
        --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.temporal_model import TemporalClassifier, WINDOW_SIZE, FEATURE_DIM
from ml.training.train_classifier import FeatureWindowDataset, CLASS_NAMES


def evaluate_model(
    checkpoint_path: str,
    data_root: str,
    device_str: str = "cpu",
    batch_size: int = 32,
) -> Dict:
    device = torch.device(device_str)

    # Load model
    model = TemporalClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"Model loaded: {checkpoint_path}")
    print(f"Checkpoint val F1: {checkpoint.get('best_val_f1', 'N/A')}")

    # Dataset
    dataset = FeatureWindowDataset(data_root, augment=False)
    if len(dataset) == 0:
        print(f"ERROR: No samples found in {data_root}")
        return {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating on {len(dataset)} samples...\n")

    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    n = len(CLASS_NAMES)
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    tn = [0] * n

    for pred, label in zip(all_preds, all_labels):
        for c in range(n):
            if pred == c and label == c:
                tp[c] += 1
            elif pred == c and label != c:
                fp[c] += 1
            elif pred != c and label == c:
                fn[c] += 1
            else:
                tn[c] += 1

    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10} {'FNR':>10}")
    print("-" * 70)
    metrics = {}
    for c, name in enumerate(CLASS_NAMES):
        precision = tp[c] / (tp[c] + fp[c] + 1e-6)
        recall = tp[c] / (tp[c] + fn[c] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        fpr = fp[c] / (fp[c] + tn[c] + 1e-6)
        fnr = fn[c] / (fn[c] + tp[c] + 1e-6)
        print(f"{name:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} "
              f"{fpr:>10.4f} {fnr:>10.4f}")
        metrics[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "fnr": fnr, "tp": tp[c], "fp": fp[c],
            "fn": fn[c], "tn": tn[c],
        }

    # Overall accuracy
    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_preds)
    mean_f1 = sum(metrics[c]["f1"] for c in CLASS_NAMES) / n

    print("-" * 70)
    print(f"{'Overall accuracy':<20} {accuracy:>10.4f}")
    print(f"{'Mean F1':<20} {mean_f1:>10.4f}")

    # Confusion matrix
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>12}", end="")
    for name in CLASS_NAMES:
        print(f"{name[:10]:>12}", end="")
    print()
    cm = defaultdict(lambda: defaultdict(int))
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1
    for actual in range(n):
        print(f"{CLASS_NAMES[actual][:10]:>12}", end="")
        for predicted in range(n):
            print(f"{cm[actual][predicted]:>12}", end="")
        print()

    metrics["_summary"] = {"accuracy": accuracy, "mean_f1": mean_f1}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmartOmniSentinel classifier")
    parser.add_argument("--checkpoint", default="ml/checkpoints/temporal_v1.pt")
    parser.add_argument("--data_root", default="data/processed/val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    evaluate_model(args.checkpoint, args.data_root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
