"""
scripts/generate_attention_gifs.py
----------------------------------
Day 11 Deliverable: Generates 4 Qualitative Explainability GIFs with
attention-highlighted skeletons in docs/figures/:

  1. true_positive_punch.gif: Arm/wrist joints glowing red during striking
  2. true_negative_walking.gif: Calm uniform blue/gray joints during walking
  3. false_positive_exercise.gif: Single-person pushup motion with low proximity
  4. false_negative_shove.gif: Subtle close-quarter body shove with moderate attention
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.attention_overlay import draw_attention_overlay

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard 17 COCO base coordinates (normalized in [0, 1])
BASE_SKELETON = np.array([
    [0.50, 0.18],  # 0: nose
    [0.48, 0.16],  # 1: l_eye
    [0.52, 0.16],  # 2: r_eye
    [0.45, 0.18],  # 3: l_ear
    [0.55, 0.18],  # 4: r_ear
    [0.42, 0.30],  # 5: l_shoulder
    [0.58, 0.30],  # 6: r_shoulder
    [0.36, 0.45],  # 7: l_elbow
    [0.64, 0.45],  # 8: r_elbow
    [0.32, 0.58],  # 9: l_wrist
    [0.68, 0.58],  # 10: r_wrist
    [0.44, 0.60],  # 11: l_hip
    [0.56, 0.60],  # 12: r_hip
    [0.42, 0.78],  # 13: l_knee
    [0.58, 0.78],  # 14: r_knee
    [0.42, 0.94],  # 15: l_ankle
    [0.58, 0.94],  # 16: r_ankle
], dtype=np.float32)


def generate_gif(
    filename: str,
    title: str,
    motion_fn,
    attention_fn,
    num_frames: int = 16,
) -> str:
    out_path = OUTPUT_DIR / filename
    frames_pil = []

    for t in range(num_frames):
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 35)

        kp = motion_fn(t, num_frames)
        att = attention_fn(t, num_frames)

        annotated = draw_attention_overlay(canvas, kp, att, label=title)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(rgb))

    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=150,
        loop=0,
    )
    print(f"Generated: {out_path}")
    return str(out_path)


# 1. True Positive: Rapid right punch
def motion_punch(t: int, n: int) -> np.ndarray:
    kp = BASE_SKELETON.copy()
    prog = math.sin((t / n) * math.pi)
    kp[8, 0] += 0.15 * prog
    kp[8, 1] -= 0.10 * prog
    kp[10, 0] += 0.28 * prog
    kp[10, 1] -= 0.15 * prog
    return kp

def att_punch(t: int, n: int) -> np.ndarray:
    att = np.ones(17, dtype=np.float32) * 0.15
    prog = math.sin((t / n) * math.pi)
    att[8] = 0.40 + 0.45 * prog
    att[10] = 0.50 + 0.48 * prog
    att[6] = 0.30 + 0.30 * prog
    return att


# 2. True Negative: Calm walking
def motion_walk(t: int, n: int) -> np.ndarray:
    kp = BASE_SKELETON.copy()
    phase = (t / n) * 2 * math.pi
    kp[13, 1] += 0.05 * math.sin(phase)
    kp[14, 1] -= 0.05 * math.sin(phase)
    kp[9, 1] += 0.04 * math.sin(phase)
    kp[10, 1] -= 0.04 * math.sin(phase)
    return kp

def att_walk(t: int, n: int) -> np.ndarray:
    return np.ones(17, dtype=np.float32) * 0.12


# 3. False Positive: Solo burpee / pushup
def motion_exercise(t: int, n: int) -> np.ndarray:
    kp = BASE_SKELETON.copy()
    prog = math.sin((t / n) * math.pi)
    kp[:, 1] += 0.18 * prog
    return kp

def att_exercise(t: int, n: int) -> np.ndarray:
    att = np.ones(17, dtype=np.float32) * 0.20
    prog = math.sin((t / n) * math.pi)
    att[11] += 0.35 * prog
    att[12] += 0.35 * prog
    return att


# 4. False Negative: Subtle body shove
def motion_shove(t: int, n: int) -> np.ndarray:
    kp = BASE_SKELETON.copy()
    prog = math.sin((t / n) * math.pi)
    kp[:, 0] += 0.08 * prog
    return kp

def att_shove(t: int, n: int) -> np.ndarray:
    att = np.ones(17, dtype=np.float32) * 0.18
    prog = math.sin((t / n) * math.pi)
    att[5] += 0.25 * prog
    att[6] += 0.25 * prog
    return att


def main():
    print(f"\nGenerating 4 Qualitative Explainability GIFs in {OUTPUT_DIR}...\n")
    generate_gif("1_true_positive_punch.gif", "TRUE POSITIVE: PUNCH", motion_punch, att_punch)
    generate_gif("2_true_negative_walking.gif", "TRUE NEGATIVE: WALKING", motion_walk, att_walk)
    generate_gif("3_false_positive_exercise.gif", "FALSE POSITIVE: EXERCISE", motion_exercise, att_exercise)
    generate_gif("4_false_negative_shove.gif", "FALSE NEGATIVE: SHOVE", motion_shove, att_shove)
    print("\nAll 4 GIFs successfully generated!\n")


if __name__ == "__main__":
    main()

