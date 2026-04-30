"""
ml/models/fall_model.py
------------------------
Binary fall detection classifier (optional companion to geometric rules).

Input: Per-frame pose feature vector
    17 keypoints (x, y) + 17 keypoint confidences = 51 features

Output: Probability of fall (binary sigmoid)

Used as the classifier branch in services/inference/fall_detector.py
to confirm geometric triggers. Without this model, fall_detector
relies on geometric rules alone (still effective).
"""
from __future__ import annotations

import torch
import torch.nn as nn

FALL_INPUT_DIM = 51   # 34 (kp coords) + 17 (kp conf)


class FallClassifier(nn.Module):
    """
    Lightweight MLP binary fall classifier.
    Takes a single frame's pose features as input.
    No temporal modeling — relies on geometric context from fall_detector.
    """

    def __init__(
        self,
        input_dim: int = FALL_INPUT_DIM,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, FALL_INPUT_DIM)
        Returns:
            logits: (batch, 1) — apply sigmoid for probability
        """
        return self.net(x)

    def predict_prob(self, x: torch.Tensor) -> float:
        """Single-sample fall probability."""
        with torch.no_grad():
            logit = self.forward(x)
            return float(torch.sigmoid(logit)[0, 0])
