"""
ml/datasets/clip_splitter.py
------------------------------
Splits long annotated video clips into fixed-duration segments.
Used when the source dataset provides long videos with time-coded annotations
rather than pre-split clips (e.g., UCF-Crime, custom recordings).

Output: One .mp4 clip per annotation span, saved to:
    data/raw/clips/{class_name}/{source}_{start}_{end}.mp4
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List

import cv2


@dataclass
class Annotation:
    video_path: str
    label: str
    start_frame: int
    end_frame: int
    notes: str = ""


def split_video_by_annotations(
    annotations: List[Annotation],
    output_root: str,
    context_frames: int = 16,
) -> int:
    """
    Extract clips defined by annotations from source videos.
    Adds context_frames before and after each annotation span.

    Returns number of clips extracted.
    """
    os.makedirs(output_root, exist_ok=True)
    extracted = 0

    for ann in annotations:
        cap = cv2.VideoCapture(ann.video_path)
        if not cap.isOpened():
            print(f"WARNING: Cannot open {ann.video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start = max(0, ann.start_frame - context_frames)
        end = ann.end_frame + context_frames

        out_dir = os.path.join(output_root, ann.label)
        os.makedirs(out_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(ann.video_path))[0]
        out_path = os.path.join(out_dir, f"{stem}_{ann.start_frame}_{ann.end_frame}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_idx in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap.release()
        extracted += 1

    return extracted


def load_annotations_from_csv(csv_path: str) -> List[Annotation]:
    """
    Load annotations from a CSV file with columns:
    video_path, label, start_frame, end_frame, notes

    Example CSV:
        /data/videos/incident_001.mp4,violent,150,320,two people fighting
        /data/videos/normal_003.mp4,normal,0,500,lobby walk-through
    """
    annotations = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(Annotation(
                video_path=row["video_path"],
                label=row["label"].strip().lower(),
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                notes=row.get("notes", ""),
            ))
    return annotations
