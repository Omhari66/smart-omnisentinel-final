"""
ml/datasets/ucfcrime_loader.py
--------------------------------
Loader for the UCF-Crime dataset.
https://www.crcv.ucf.edu/projects/real-world/

UCF-Crime contains 1900 long untrimmed surveillance videos across 13 anomaly classes.
We use a subset relevant to SmartOmniSentinel:

    Relevant → map to "violent":
        Fighting, Assault, Robbery, Shooting, Stealing (with physical contact)

    Relevant → map to "distress":
        Abuse (where victim is on ground / incapacitated)

    Ignore (out of scope for V1):
        Arson, Burglary, Shoplifting, Vandalism, Road Accidents, Explosion

    Normal:
        Normal_Videos_event/ directory

Dataset structure:
    UCF_Crimes/
        Videos/
            Fighting/    # .mp4 files (long, untrimmed)
            Assault/
            Abuse/
            Normal_Videos_event/
            ...
        Annotations/
            Temporal_Anomaly_Annotation.txt  # Format: video,start_frame,end_frame,label

Strategy for extraction:
  1. Use Temporal_Anomaly_Annotation.txt to find anomaly segments
  2. Extract only those segments + context frames
  3. Run pose extraction on sampled frames
  4. Build (16, 55) feature windows

Note: UCF-Crime videos are mostly CCTV-quality — good for domain adaptation.
The main challenge is label ambiguity (some "fights" are staged).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Mapping from UCF-Crime class names to our training labels
UCF_CLASS_MAP: Dict[str, Optional[str]] = {
    "Fighting": "violent",
    "Assault": "violent",
    "Robbery": "violent",
    "Shooting": "violent",
    "Abuse": "distress",
    "Normal_Videos_event": "normal",
    # Classes we skip (out of V1 scope):
    "Arson": None,
    "Burglary": None,
    "Shoplifting": None,
    "Vandalism": None,
    "RoadAccidents": None,
    "Explosion": None,
    "Arrest": None,
}


@dataclass
class UCFAnnotation:
    video_path: str
    label: str          # Our mapped label
    start_frame: int
    end_frame: int
    ucf_class: str      # Original UCF class name


def parse_temporal_annotations(annotation_file: str, videos_root: str) -> List[UCFAnnotation]:
    """
    Parse UCF-Crime temporal annotation file.

    Expected format (tab-separated):
        video_name  start1  end1  start2  end2
    or:
        video_name  -1  -1  (for normal videos with no anomaly)

    Returns list of UCFAnnotation with resolved file paths.
    """
    annotations = []

    with open(annotation_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            video_name = parts[0]
            # Extract class from video name prefix (e.g. "Fighting001_x264.mp4" → "Fighting")
            ucf_class = _extract_class_from_name(video_name)
            label = UCF_CLASS_MAP.get(ucf_class)
            if label is None:
                continue  # Skip irrelevant classes

            # Try to find the video file
            video_path = _find_video(videos_root, ucf_class, video_name)
            if video_path is None:
                continue

            # Parse start/end frames
            try:
                if parts[1] == "-1":
                    # Normal video — use first 300 frames as normal window
                    annotations.append(UCFAnnotation(
                        video_path=video_path,
                        label="normal",
                        start_frame=0,
                        end_frame=300,
                        ucf_class=ucf_class,
                    ))
                else:
                    start = int(parts[1])
                    end = int(parts[2])
                    annotations.append(UCFAnnotation(
                        video_path=video_path,
                        label=label,
                        start_frame=max(0, start - 16),  # Pre-context
                        end_frame=end + 16,              # Post-context
                        ucf_class=ucf_class,
                    ))
            except (ValueError, IndexError):
                continue

    return annotations


def _extract_class_from_name(video_name: str) -> str:
    """Extract UCF class prefix from video filename."""
    # e.g. "Fighting007_x264.mp4" → "Fighting"
    for cls in UCF_CLASS_MAP:
        if video_name.startswith(cls):
            return cls
    return "Unknown"


def _find_video(videos_root: str, ucf_class: str, video_name: str) -> Optional[str]:
    """Locate video file in the dataset directory."""
    # Try class subdirectory first
    path = os.path.join(videos_root, ucf_class, video_name)
    if os.path.exists(path):
        return path
    # Try root directly
    path = os.path.join(videos_root, video_name)
    if os.path.exists(path):
        return path
    return None


def process_ucfcrime_dataset(
    dataset_root: str,
    output_root: str,
    annotation_file: Optional[str] = None,
    val_fraction: float = 0.2,
    window_size: int = 16,
    feature_dim: int = 55,
    target_fps: int = 4,
) -> None:
    """
    Process UCF-Crime dataset: extract pose features and save .npy windows.

    Args:
        dataset_root:     Path to UCF_Crimes/ root
        output_root:      Output for data/processed/
        annotation_file:  Path to Temporal_Anomaly_Annotation.txt
                          If None, falls back to scanning class directories
        val_fraction:     Fraction of data for validation
        window_size:      Temporal window length
        feature_dim:      Feature vector dimension
        target_fps:       Effective FPS for frame sampling
    """
    import random
    import cv2

    if annotation_file is None:
        annotation_file = os.path.join(
            dataset_root, "Annotations", "Temporal_Anomaly_Annotation.txt"
        )

    videos_root = os.path.join(dataset_root, "Videos")

    if os.path.exists(annotation_file):
        annotations = parse_temporal_annotations(annotation_file, videos_root)
        print(f"Loaded {len(annotations)} annotated segments from annotation file")
    else:
        print(f"Annotation file not found: {annotation_file}")
        print("Falling back to class directory scan (no temporal boundaries)")
        annotations = _scan_class_directories(videos_root)

    print(f"Processing {len(annotations)} video segments...")

    # Load pose model
    try:
        from ultralytics import YOLO
        pose_model = YOLO("yolov8n-pose.pt")
    except Exception:
        pose_model = None
        print("WARNING: No pose model — features will be zeros")

    # Train/val split by video (not by frame — prevent data leakage)
    random.seed(42)
    random.shuffle(annotations)
    n_val = max(1, int(len(annotations) * val_fraction))
    val_indices = set(range(len(annotations) - n_val, len(annotations)))

    total_windows = 0
    for idx, ann in enumerate(annotations):
        split = "val" if idx in val_indices else "train"
        out_dir = os.path.join(output_root, split, ann.label)
        os.makedirs(out_dir, exist_ok=True)

        windows = _extract_windows_from_segment(
            video_path=ann.video_path,
            start_frame=ann.start_frame,
            end_frame=ann.end_frame,
            pose_model=pose_model,
            target_fps=target_fps,
            window_size=window_size,
            feature_dim=feature_dim,
        )

        stem = Path(ann.video_path).stem
        for w_idx, window in enumerate(windows):
            out_path = os.path.join(out_dir, f"{stem}_{ann.start_frame}_{w_idx:04d}.npy")
            np.save(out_path, window)
            total_windows += 1

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(annotations)} segments processed | "
                  f"Windows: {total_windows}")

    print(f"\n✓ Total windows extracted: {total_windows}")
    _print_class_distribution(output_root)


def _extract_windows_from_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    pose_model,
    target_fps: int,
    window_size: int,
    feature_dim: int,
) -> List[np.ndarray]:
    """Extract feature windows from a video segment."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, round(source_fps / target_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    features_list = []
    prev_gray = None
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - start_frame) % skip != 0:
            continue

        feat = np.zeros(feature_dim, dtype=np.float32)
        if pose_model:
            try:
                res = pose_model.predict(frame, verbose=False)
                if res and res[0].keypoints is not None and len(res[0].keypoints):
                    kp = res[0].keypoints.xyn[0].cpu().numpy()
                    conf = res[0].keypoints.conf[0].cpu().numpy()
                    feat[0:34] = kp.flatten()
                    feat[34:51] = conf
            except Exception:
                pass

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        if prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                feat[51] = min(mag / 100.0, 1.0)
                feat[52] = min(mag / 10.0, 1.0)
            except Exception:
                pass
        prev_gray = gray
        features_list.append(feat)

    cap.release()

    if len(features_list) < window_size:
        return []

    windows = []
    stride = max(1, window_size // 2)
    for i in range(0, len(features_list) - window_size + 1, stride):
        arr = np.array(features_list[i:i + window_size], dtype=np.float32)
        if arr.shape == (window_size, feature_dim):
            windows.append(arr)
    return windows


def _scan_class_directories(videos_root: str) -> List[UCFAnnotation]:
    """Fallback: scan class directories without temporal annotations."""
    annotations = []
    for cls_name, label in UCF_CLASS_MAP.items():
        if label is None:
            continue
        cls_dir = os.path.join(videos_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".mp4", ".avi")):
                annotations.append(UCFAnnotation(
                    video_path=os.path.join(cls_dir, fname),
                    label=label,
                    start_frame=0,
                    end_frame=99999,
                    ucf_class=cls_name,
                ))
    return annotations


def _print_class_distribution(output_root: str) -> None:
    """Print dataset class distribution after processing."""
    from collections import Counter
    for split in ("train", "val"):
        counts: Counter = Counter()
        split_dir = os.path.join(output_root, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                counts[cls] = len([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
        print(f"\n{split.upper()} distribution: {dict(counts)}")
