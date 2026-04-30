"""
ml/training/dataset_builder.py
---------------------------------
Unified multi-dataset builder.
Combines RWF-2000, UCF-Crime, UR Fall, and custom annotations
into a single processed feature dataset ready for training.

Usage:
    python -m ml.training.dataset_builder \
        --config configs/dataset_config.yaml \
        --output data/processed

Config example (dataset_config.yaml):
    datasets:
      - name: rwf2000
        root: data/raw/RWF-2000
        splits: [train, val]
      - name: ucfcrime
        root: data/raw/UCF_Crimes
        annotation_file: data/raw/UCF_Crimes/Annotations/Temporal_Anomaly_Annotation.txt
      - name: custom
        annotations_csv: data/annotations/custom_incidents.csv

    output_root: data/processed
    val_fraction: 0.2
    window_size: 16
    feature_dim: 55
    target_fps: 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def build_dataset_from_config(config: Dict[str, Any]) -> None:
    """Build training dataset from configuration dict."""
    output_root = config.get("output_root", "data/processed")
    window_size = config.get("window_size", 16)
    feature_dim = config.get("feature_dim", 55)
    target_fps = config.get("target_fps", 4)
    val_fraction = config.get("val_fraction", 0.2)

    os.makedirs(output_root, exist_ok=True)

    for ds_config in config.get("datasets", []):
        name = ds_config.get("name", "unknown")
        print(f"\n{'='*60}")
        print(f"Processing dataset: {name.upper()}")
        print(f"{'='*60}")

        if name == "rwf2000":
            _process_rwf2000(ds_config, output_root, window_size, feature_dim, target_fps)
        elif name == "ucfcrime":
            _process_ucfcrime(ds_config, output_root, window_size, feature_dim,
                              target_fps, val_fraction)
        elif name == "urfall":
            _process_urfall(ds_config, output_root, window_size, feature_dim)
        elif name == "custom":
            _process_custom(ds_config, output_root, window_size, feature_dim, val_fraction)
        else:
            print(f"WARNING: Unknown dataset '{name}', skipping.")

    # Final statistics
    print(f"\n{'='*60}")
    print("FINAL DATASET STATISTICS")
    print(f"{'='*60}")
    _print_full_stats(output_root)
    _check_class_balance(output_root)


def _process_rwf2000(
    config: Dict, output_root: str,
    window_size: int, feature_dim: int, target_fps: int
) -> None:
    from ml.datasets.rwf2000_loader import process_dataset
    process_dataset(
        input_root=config["root"],
        output_root=output_root,
        splits=config.get("splits", ["train", "val"]),
    )


def _process_ucfcrime(
    config: Dict, output_root: str,
    window_size: int, feature_dim: int, target_fps: int, val_fraction: float
) -> None:
    from ml.datasets.ucfcrime_loader import process_ucfcrime_dataset
    process_ucfcrime_dataset(
        dataset_root=config["root"],
        output_root=output_root,
        annotation_file=config.get("annotation_file"),
        val_fraction=val_fraction,
        window_size=window_size,
        feature_dim=feature_dim,
        target_fps=target_fps,
    )


def _process_urfall(
    config: Dict, output_root: str,
    window_size: int, feature_dim: int
) -> None:
    from ml.datasets.urfall_loader import process_urfall_dataset
    process_urfall_dataset(
        input_root=config["root"],
        output_root=output_root,
        window_size=window_size,
    )


def _process_custom(
    config: Dict, output_root: str,
    window_size: int, feature_dim: int, val_fraction: float
) -> None:
    from ml.datasets.custom_loader import process_custom_dataset
    process_custom_dataset(
        annotations_csv=config["annotations_csv"],
        output_root=output_root,
        val_fraction=val_fraction,
        window_size=window_size,
        feature_dim=feature_dim,
    )


def _print_full_stats(output_root: str) -> None:
    """Print per-split per-class sample counts."""
    for split in ("train", "val"):
        split_dir = os.path.join(output_root, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"\n{split.upper()}:")
        total = 0
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            count = len([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
            print(f"  {cls:<25} {count:>6} samples")
            total += count
        print(f"  {'TOTAL':<25} {total:>6} samples")


def _check_class_balance(output_root: str) -> None:
    """Warn if any class has less than 100 samples or is severely imbalanced."""
    print("\nClass Balance Check:")
    for split in ("train", "val"):
        split_dir = os.path.join(output_root, split)
        if not os.path.isdir(split_dir):
            continue
        counts = {}
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                counts[cls] = len([f for f in os.listdir(cls_dir) if f.endswith(".npy")])

        if not counts:
            continue

        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / max(min_count, 1)

        for cls, count in counts.items():
            status = "OK"
            if count < 100:
                status = "⚠  LOW SAMPLE COUNT"
            elif ratio > 10:
                status = "⚠  IMBALANCED"
            print(f"  [{split}] {cls}: {count} samples {status}")

        if ratio > 10:
            print(f"  ⚠  Max imbalance ratio: {ratio:.1f}x — use WeightedRandomSampler")
        else:
            print(f"  ✓ Balance ratio: {ratio:.1f}x — acceptable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SmartOmniSentinel training dataset"
    )
    parser.add_argument(
        "--config", default="configs/dataset_config.yaml",
        help="Path to dataset configuration YAML"
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="Output directory for processed .npy files"
    )
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        print("Using minimal default config (no datasets configured).")
        config = {"output_root": args.output, "datasets": []}

    config["output_root"] = args.output
    build_dataset_from_config(config)


if __name__ == "__main__":
    main()
