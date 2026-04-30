"""
ml/datasets/custom_loader.py
------------------------------
Loader for custom-annotated CCTV footage.
Use this when building your own training data from site-specific recordings.

Annotation format: CSV with columns:
    video_path, label, start_frame, end_frame, notes

Supported labels: normal, violent, distress, fall, crowd_aggression

Steps:
    1. Record relevant CCTV clips
    2. Annotate using any video annotation tool (CVAT, LabelStudio, VGG VIA)
    3. Export annotations as CSV in the format above
    4. Run this loader to extract feature windows
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class CustomAnnotation:
    video_path: str
    label: str       # "normal" | "violent" | "distress" | "fall" | "crowd_aggression"
    start_frame: int
    end_frame: int
    notes: str = ""


# Map custom labels to training class names
LABEL_MAP = {
    "normal": "normal",
    "nonviolent": "normal",
    "fight": "violent",
    "violent": "violent",
    "violence": "violent",
    "assault": "violent",
    "fall": "fall",
    "collapse": "fall",
    "distress": "distress",
    "crowd": "crowd_aggression",
    "panic": "crowd_aggression",
    "crowd_aggression": "crowd_aggression",
}


def load_annotations_csv(csv_path: str) -> List[CustomAnnotation]:
    """Load annotations from CSV file."""
    annotations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = LABEL_MAP.get(row["label"].strip().lower(), "normal")
            annotations.append(CustomAnnotation(
                video_path=row["video_path"].strip(),
                label=label,
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                notes=row.get("notes", ""),
            ))
    return annotations


def process_custom_dataset(
    annotations_csv: str,
    output_root: str,
    val_fraction: float = 0.2,
    window_size: int = 16,
    feature_dim: int = 55,
) -> None:
    """
    Extract feature windows from custom annotations.
    Auto-splits into train/val.
    """
    import random
    import cv2

    annotations = load_annotations_csv(annotations_csv)
    print(f"Loaded {len(annotations)} annotations from {annotations_csv}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(annotations)
    n_val = max(1, int(len(annotations) * val_fraction))
    val_set = set(range(len(annotations) - n_val, len(annotations)))

    try:
        from ultralytics import YOLO
        pose_model = YOLO("yolov8n-pose.pt")
    except Exception:
        pose_model = None
        print("WARNING: No pose model — features will be zeros")

    total_windows = 0
    for idx, ann in enumerate(annotations):
        split = "val" if idx in val_set else "train"
        out_dir = os.path.join(output_root, split, ann.label)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(ann.video_path)
        if not cap.isOpened():
            print(f"  SKIP: Cannot open {ann.video_path}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, ann.start_frame)
        features_list = []
        prev_gray = None

        for frame_idx in range(ann.start_frame, ann.end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            features = np.zeros(feature_dim, dtype=np.float32)
            if pose_model:
                try:
                    results = pose_model.predict(frame, verbose=False)
                    if results and results[0].keypoints is not None and \
                       len(results[0].keypoints) > 0:
                        kp = results[0].keypoints.xyn[0].cpu().numpy()
                        conf = results[0].keypoints.conf[0].cpu().numpy()
                        features[0:34] = kp.flatten()
                        features[34:51] = conf
                except Exception:
                    pass

            # Optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 120))
            if prev_gray is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                    features[51] = min(mag / 100.0, 1.0)
                    features[52] = min(mag / 10.0, 1.0)
                except Exception:
                    pass
            prev_gray = gray
            features_list.append(features)

        cap.release()

        # Build windows
        stem = Path(ann.video_path).stem
        for i in range(0, max(1, len(features_list) - window_size + 1), window_size // 2):
            window = features_list[i:i + window_size]
            if len(window) < window_size:
                window = window + [window[-1]] * (window_size - len(window))
            arr = np.array(window[:window_size], dtype=np.float32)
            out_path = os.path.join(out_dir, f"{stem}_{ann.start_frame}_{i:04d}.npy")
            np.save(out_path, arr)
            total_windows += 1

        print(f"  [{split}] {ann.label}: {ann.video_path} → "
              f"{len(features_list)} frames, windows saved")

    print(f"\n✓ Total windows extracted: {total_windows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Path to annotations CSV")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    args = parser.parse_args()
    process_custom_dataset(args.annotations, args.output, args.val_fraction)
