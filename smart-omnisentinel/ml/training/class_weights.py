"""
ml/training/class_weights.py
------------------------------
Utilities for computing class weights to handle dataset imbalance.

Violence/fall datasets are always imbalanced: normal events dominate.
Without correction, the model learns to always predict "normal".

Two strategies implemented:
  1. Inverse-frequency weights → passed to nn.CrossEntropyLoss(weight=...)
  2. WeightedRandomSampler   → oversamples minority classes per epoch
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import torch


def compute_class_weights_inverse_freq(
    labels: List[int],
    n_classes: int,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    Higher weight = rarer class = larger loss gradient for that class.

    Args:
        labels: List of integer class labels from the dataset
        n_classes: Total number of classes
        smoothing: Add to all counts to avoid division by zero

    Returns:
        Tensor of shape (n_classes,) with normalized weights
    """
    counts = Counter(labels)
    total = len(labels)

    weights = []
    for c in range(n_classes):
        count = counts.get(c, 0) + smoothing
        weights.append(total / count)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize so mean weight = 1
    weight_tensor = weight_tensor / weight_tensor.mean()
    return weight_tensor


def compute_sample_weights(
    labels: List[int],
    class_weights: torch.Tensor,
) -> List[float]:
    """
    Compute per-sample weights for WeightedRandomSampler.
    Each sample's weight = its class's inverse-frequency weight.
    """
    return [float(class_weights[label]) for label in labels]


def print_class_distribution(labels: List[int], class_names: List[str]) -> None:
    """Print class distribution for dataset analysis."""
    counts = Counter(labels)
    total = len(labels)
    print(f"\nClass Distribution ({total} total samples):")
    print(f"{'Class':<20} {'Count':>8} {'Pct':>8} {'Weight':>10}")
    print("-" * 50)
    weights = compute_class_weights_inverse_freq(labels, len(class_names))
    for i, name in enumerate(class_names):
        count = counts.get(i, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"{name:<20} {count:>8} {pct:>7.1f}% {weights[i]:>10.3f}x")
    print()
