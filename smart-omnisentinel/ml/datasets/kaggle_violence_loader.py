import os
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import argparse

# YOLO pose model
pose_model = YOLO("yolov8n-pose.pt")

def extract_keypoints(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while len(sequence) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kp = results[0].keypoints.xy[0].cpu().numpy()

            # Ensure shape is (17, 2)
            if kp.shape[0] == 17:
                keypoints = kp.flatten()  # → (34,)
            else:
                keypoints = np.zeros(34)
        else:
            keypoints = np.zeros(34)

        # FORCE FIXED SIZE
        if keypoints.shape[0] != 34:
            keypoints = np.zeros(34)

        sequence.append(keypoints)

    cap.release()

    if len(sequence) < max_frames:
        return None

    return np.stack(sequence)  # safer than np.array

def process_dataset(input_dir, output_dir, max_videos=100):
    classes = {
        "Violence": "violent",
        "NonViolence": "normal"
    }

    for class_name, label in classes.items():
        class_path = Path(input_dir) / class_name
        videos = list(class_path.glob("*.mp4"))[:max_videos]

        print(f"\nProcessing {class_name} → {label}")

        for i, video in enumerate(videos):
            seq = extract_keypoints(str(video))

            if seq is None:
                continue

            save_dir = Path(output_dir) / "train" / label
            save_dir.mkdir(parents=True, exist_ok=True)

            np.save(save_dir / f"{video.stem}.npy", seq)

            print(f"[{i+1}/{len(videos)}] Saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_videos", type=int, default=100)

    args = parser.parse_args()

    process_dataset(args.input, args.output, args.max_videos)