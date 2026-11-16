"""
visualization/attention_overlay.py
----------------------------------
Day 11 Deliverable: Explainable Joint Attention Skeleton Renderer.

Visualizes model explainability:
  - High-contribution joints (e.g. punching wrist, striking elbow): Red, large glowing dot
  - Low-contribution joints (e.g. passive ankle): Gray, small dot
  - Kinematic bone edges: Colormapped and scaled by attention weight
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.models.st_gcn import COCO_BONES, NUM_COCO_JOINTS


def draw_attention_overlay(
    frame: np.ndarray,
    keypoints: np.ndarray,
    joint_attention: np.ndarray,
    label: str = "VIOLENT PUNCH ATTENTION",
) -> np.ndarray:
    """
    Renders explainable attention weights onto an image frame.

    Args:
        frame: (H, W, 3) BGR image
        keypoints: (17, 2) in [0, 1] normalized or pixel coordinates
        joint_attention: (17,) values in [0, 1]
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()

    pts = keypoints.copy()
    if np.max(pts) <= 1.5:
        pts[:, 0] *= w
        pts[:, 1] *= h

    # 1. Draw attention-weighted bone edges
    for i, j in COCO_BONES:
        p1 = (int(pts[i, 0]), int(pts[i, 1]))
        p2 = (int(pts[j, 0]), int(pts[j, 1]))
        if p1 == (0, 0) or p2 == (0, 0):
            continue

        edge_att = float((joint_attention[i] + joint_attention[j]) / 2.0)

        if edge_att > 0.6:
            # Violent trigger edge (Red / Orange)
            color = (0, 69, 255)
            thickness = 4
        elif edge_att > 0.3:
            # Moderate edge (Yellow)
            color = (0, 215, 255)
            thickness = 2
        else:
            # Low contribution edge (Gray)
            color = (128, 128, 128)
            thickness = 1

        cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    # 2. Draw joints with dot size & color proportional to saliency
    for idx in range(NUM_COCO_JOINTS):
        pt = (int(pts[idx, 0]), int(pts[idx, 1]))
        if pt == (0, 0):
            continue

        att = float(joint_attention[idx])
        if att > 0.6:
            # Saliency hotspot
            radius = int(6 + 8 * att)
            color = (0, 0, 255)  # Bright Red
            cv2.circle(canvas, pt, radius + 2, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        elif att > 0.3:
            radius = int(4 + 4 * att)
            color = (0, 255, 255)  # Yellow
        else:
            radius = 3
            color = (160, 160, 160)  # Gray

        cv2.circle(canvas, pt, radius, color, -1, lineType=cv2.LINE_AA)

    # 3. Informational HUD
    top_joint_idx = int(np.argmax(joint_attention))
    joint_names = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"
    ]
    name = joint_names[top_joint_idx] if top_joint_idx < len(joint_names) else f"joint_{top_joint_idx}"
    hud_text = f"Top Attention: {name.upper()} ({joint_attention[top_joint_idx]:.2f}) | {label}"
    cv2.putText(canvas, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return canvas
