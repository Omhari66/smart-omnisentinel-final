"""
ml/datasets/urfall_loader.py
------------------------------
Loader for the UR Fall Detection dataset.
http://fenix.ur.edu.pl/~mkepski/ds/uf.html

Dataset contains depth + RGB sequences of falls vs. Activities of Daily Living (ADL).
We use the RGB sequences for pose-based feature extraction.

Structure:
    urfall/
        fall-XX-cam0-rgb/   # Fall sequences (directories of JPEG frames)
        adl-XX-cam0-rgb/    # Normal activities
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np


def process_urfall_dataset(
    input_root: str,
    output_root: str,
    splits: dict = None,  # {"train": 0.8, "val": 0.2}
    window_size: int = 16,
    feature_dim: int = 51,  # 17 keypoints (x,y) + 17 conf = 51 for fall model
) -> None:
    """
    Process UR Fall dataset for the binary fall classifier.
    Saves (window_size, feature_dim) .npy files.

    Args:
        input_root: Root directory of urfall dataset
        output_root: Output for processed data
        splits: Train/val split ratios
    """
    if splits is None:
        splits = {"train": 0.8, "val": 0.2}

    try:
        from ultralytics import YOLO
        pose_model = YOLO("yolov8n-pose.pt")
    except Exception:
        pose_model = None
        print("WARNING: Running without pose model — features will be zeros")

    import cv2

    fall_dirs = sorted([
        d for d in os.listdir(input_root)
        if d.startswith("fall-") and os.path.isdir(os.path.join(input_root, d))
    ])
    adl_dirs = sorted([
        d for d in os.listdir(input_root)
        if d.startswith("adl-") and os.path.isdir(os.path.join(input_root, d))
    ])

    def process_sequence(seq_dir: str, label: str, split: str) -> int:
        """Process one image sequence directory → .npy windows."""
        frame_dir = os.path.join(input_root, seq_dir)
        frames = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])
        if not frames:
            return 0

        features_list = []
        for frame_file in frames:
            img = cv2.imread(os.path.join(frame_dir, frame_file))
            if img is None:
                continue
            features = np.zeros(feature_dim, dtype=np.float32)
            if pose_model:
                try:
                    results = pose_model.predict(img, verbose=False)
                    if results and results[0].keypoints is not None and len(results[0].keypoints):
                        kp = results[0].keypoints.xyn[0].cpu().numpy()
                        conf = results[0].keypoints.conf[0].cpu().numpy()
                        features[0:34] = kp.flatten()
                        features[34:51] = conf
                except Exception:
                    pass
            features_list.append(features)

        if len(features_list) < window_size:
            return 0

        out_dir = os.path.join(output_root, split, label)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        for i in range(0, len(features_list) - window_size + 1, window_size // 2):
            window = np.array(features_list[i:i + window_size], dtype=np.float32)
            if window.shape == (window_size, feature_dim):
                np.save(os.path.join(out_dir, f"{seq_dir}_w{i:04d}.npy"), window)
                saved += 1
        return saved

    # Split sequences into train/val
    import random
    random.seed(42)
    all_falls = list(fall_dirs)
    all_adls = list(adl_dirs)
    random.shuffle(all_falls)
    random.shuffle(all_adls)

    n_fall_train = int(len(all_falls) * splits["train"])
    n_adl_train = int(len(all_adls) * splits["train"])

    split_map = {}
    for d in all_falls[:n_fall_train]:
        split_map[d] = ("train", "fall")
    for d in all_falls[n_fall_train:]:
        split_map[d] = ("val", "fall")
    for d in all_adls[:n_adl_train]:
        split_map[d] = ("train", "normal")
    for d in all_adls[n_adl_train:]:
        split_map[d] = ("val", "normal")

    total = 0
    for seq_dir, (split, label) in split_map.items():
        n = process_sequence(seq_dir, label, split)
        total += n
        print(f"  {seq_dir} ({split}/{label}): {n} windows")

    print(f"\n✓ Total windows: {total}")
