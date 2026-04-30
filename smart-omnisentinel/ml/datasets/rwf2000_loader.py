"""
ml/datasets/rwf2000_loader.py
-------------------------------
Loader for the RWF-2000 violence detection dataset.
https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

Dataset structure:
    RWF-2000/
        train/
            Fight/      # .avi clips of violent interactions
            NonFight/   # .avi clips of normal activity
        val/
            Fight/
            NonFight/

This loader:
1. Reads video clips
2. Extracts pose features using YOLOv8n-pose at 4fps effective
3. Builds (16, 55) sliding window feature arrays
4. Saves as .npy files in the processed data directory
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_windows_from_video(
    video_path: str,
    window_size: int = 16,
    feature_dim: int = 55,
    target_fps: int = 4,
    pose_model=None,
    detector_model=None,
) -> List[np.ndarray]:
    """
    Extract sliding window feature arrays from a video clip.
    Returns list of (window_size, feature_dim) arrays.
    One window per video clip for short clips; multiple for long clips.
    """
    import cv2
    windows = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return windows

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, round(source_fps / target_fps))

    frame_features = []
    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        features = _extract_frame_features(frame, pose_model, detector_model, prev_gray)
        frame_features.append(features)

        # Compute gray for optical flow
        import cv2 as _cv2
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.resize(gray, (160, 120))
        prev_gray = gray

    cap.release()

    # Build sliding windows (stride = window_size for non-overlapping)
    stride = max(1, window_size // 2)
    for start in range(0, max(1, len(frame_features) - window_size + 1), stride):
        window = frame_features[start: start + window_size]
        if len(window) < window_size:
            # Pad with last frame features
            window = window + [window[-1]] * (window_size - len(window))
        arr = np.array(window[:window_size], dtype=np.float32)
        if arr.shape == (window_size, feature_dim):
            windows.append(arr)

    return windows


def _extract_frame_features(frame, pose_model, detector_model, prev_gray) -> np.ndarray:
    """
    Extract (55,) feature vector from one frame.
    Falls back to zeros if models not loaded (for offline test).
    """
    import cv2
    features = np.zeros(55, dtype=np.float32)

    if pose_model is None:
        return features

    try:
        results = pose_model.predict(frame, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kp = results[0].keypoints.xyn[0].cpu().numpy()   # (17, 2)
            conf = results[0].keypoints.conf[0].cpu().numpy() # (17,)
            features[0:34] = kp.flatten()
            features[34:51] = conf

        # Motion from optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
            features[51] = min(mag / 100.0, 1.0)
            features[52] = min(mag / 10.0, 1.0)
    except Exception:
        pass

    return features


def process_dataset(
    input_root: str,
    output_root: str,
    splits: List[str] = None,
) -> None:
    """
    Process full RWF-2000 dataset: extract features and save .npy windows.

    Args:
        input_root: Path to RWF-2000/ root directory
        output_root: Path to data/processed/ output directory
        splits: ["train", "val"] — which splits to process
    """
    if splits is None:
        splits = ["train", "val"]

    print("Loading pose model for feature extraction...")
    try:
        from ultralytics import YOLO
        pose_model = YOLO("yolov8n-pose.pt")
        print("Pose model loaded.")
    except Exception:
        pose_model = None
        print("WARNING: Pose model not available. Features will be zeros.")

    class_map = {"Fight": "violent", "NonFight": "normal"}

    for split in splits:
        for src_class, dst_class in class_map.items():
            src_dir = os.path.join(input_root, split, src_class)
            dst_dir = os.path.join(output_root, split, dst_class)
            os.makedirs(dst_dir, exist_ok=True)

            if not os.path.exists(src_dir):
                print(f"  Skipping {src_dir} (not found)")
                continue

            videos = [f for f in os.listdir(src_dir)
                      if f.lower().endswith((".avi", ".mp4", ".mov"))]
            print(f"Processing {split}/{src_class} → {dst_class}: {len(videos)} videos")

            saved = 0
            for i, video_file in enumerate(videos):
                video_path = os.path.join(src_dir, video_file)
                windows = extract_windows_from_video(
                    video_path=video_path,
                    pose_model=pose_model,
                )
                for j, window in enumerate(windows):
                    stem = os.path.splitext(video_file)[0]
                    out_path = os.path.join(dst_dir, f"{stem}_w{j:03d}.npy")
                    np.save(out_path, window)
                    saved += 1

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(videos)} videos processed, {saved} windows saved")

            print(f"  ✓ {split}/{src_class}: {saved} windows saved to {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="RWF-2000 dataset root")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.splits)
