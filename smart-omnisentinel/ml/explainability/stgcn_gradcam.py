"""
ml/explainability/stgcn_gradcam.py
----------------------------------
Spatial-Temporal Grad-CAM & Joint Attention Visualizer for ST-GCN.

Computes joint-level saliency heatmaps to explain WHICH joints and WHICH
time steps triggered the violence classification (e.g., highlighting the
punching wrist and striking elbow in bright red).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from ml.models.st_gcn import COCO_BONES, NUM_COCO_JOINTS, STGCNClassifier


class STGCNGradCAM:
    """
    Grad-CAM for Spatio-Temporal Graph Convolutional Networks.
    """

    def __init__(self, model: STGCNClassifier):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self.model.block4

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate_heatmap(
        self,
        x: torch.Tensor,
        proximity: Optional[torch.Tensor] = None,
        target_class: int = 1,  # 1 = Violent Interaction
    ) -> np.ndarray:
        """
        Computes normalized joint attention weights across all 17 COCO joints.
        Returns:
            joint_weights: np.ndarray of shape (17,), values in [0.0, 1.0]
        """
        self.model.eval()
        self.model.zero_grad()

        # Enable gradient computation for Grad-CAM
        x = x.clone().requires_grad_(True)
        logits = self.model(x, proximity)

        score = logits[:, target_class].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            # Fallback uniform attention if hooks failed
            return np.ones(NUM_COCO_JOINTS, dtype=np.float32) / NUM_COCO_JOINTS

        # Activations shape: (batch, channels=128, time, joints=17)
        # Gradients shape:   (batch, channels=128, time, joints=17)
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
        cam = torch.sum(alpha * self.activations, dim=1)  # (batch, time, joints)
        cam = torch.relu(cam)  # Saliency only for positive evidence

        # Mean across temporal dimension to obtain per-joint importance
        joint_importance = torch.mean(cam, dim=1)[0].cpu().numpy()  # (17,)

        # Min-max normalization
        min_val = np.min(joint_importance)
        max_val = np.max(joint_importance)
        if max_val - min_val > 1e-6:
            joint_importance = (joint_importance - min_val) / (max_val - min_val)
        else:
            joint_importance = np.zeros_like(joint_importance)

        return joint_importance

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def render_attention_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    joint_weights: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Renders an explainable attention-colored skeleton onto an OpenCV BGR frame.

    Args:
        frame: OpenCV BGR image (H, W, 3)
        keypoints: Normalized [0, 1] or pixel coordinates (17, 2)
        joint_weights: Normalized attention scores [0, 1] for all 17 joints
    Returns:
        annotated_frame: OpenCV BGR frame with heatmapped skeleton
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    # Denormalize keypoints if in [0, 1]
    pts = keypoints.copy()
    if np.max(pts) <= 1.5:
        pts[:, 0] *= w
        pts[:, 1] *= h

    # Draw bones
    for i, j in COCO_BONES:
        p1 = (int(pts[i, 0]), int(pts[i, 1]))
        p2 = (int(pts[j, 0]), int(pts[j, 1]))
        if p1 == (0, 0) or p2 == (0, 0):
            continue

        # Bone weight is average of both joint weights
        bone_weight = float((joint_weights[i] + joint_weights[j]) / 2.0)
        # Colormap: Blue (cold/calm) -> Yellow -> Red (violent trigger)
        b = int(255 * (1.0 - bone_weight))
        r = int(255 * bone_weight)
        g = int(255 * (1.0 - abs(bone_weight - 0.5) * 2))
        color = (b, g, r)

        thickness = 2 if bone_weight < 0.5 else 4
        cv2.line(out, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    # Draw joints with glowing radius proportional to attention score
    for i in range(NUM_COCO_JOINTS):
        pt = (int(pts[i, 0]), int(pts[i, 1]))
        if pt == (0, 0):
            continue

        weight = float(joint_weights[i])
        b = int(255 * (1.0 - weight))
        r = int(255 * weight)
        g = int(255 * (1.0 - abs(weight - 0.5) * 2))
        color = (b, g, r)

        radius = int(4 + 8 * weight)
        cv2.circle(out, pt, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, pt, radius + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return out
