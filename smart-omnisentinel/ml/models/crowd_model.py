"""
ml/models/crowd_model.py
--------------------------
LSTM-based crowd aggression detector (V2 — not required for MVP).

Input: Sequence of aggregate scene features
    (people_count, flow_magnitude, direction_variance, centroid_spread)
    Shape: (batch, SEQUENCE_LEN=32, 4)

Output: Binary classification — crowd_aggression (1) or normal (0)

In V1, the InferenceRunner uses a heuristic crowd detector (crowd_analyzer.py).
This model replaces the heuristic when trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SEQUENCE_LEN = 32
CROWD_FEATURE_DIM = 4
CROWD_HIDDEN_DIM = 64


class CrowdAggressionLSTM(nn.Module):
    """
    Lightweight LSTM crowd aggression detector.
    Designed to run in real time — small hidden size.
    """

    def __init__(
        self,
        input_dim: int = CROWD_FEATURE_DIM,
        hidden_dim: int = CROWD_HIDDEN_DIM,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, SEQUENCE_LEN, CROWD_FEATURE_DIM)
        Returns:
            logits: (batch, 1) — apply sigmoid for probability
        """
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.classifier(last)

    def predict_prob(self, x: torch.Tensor) -> float:
        """Convenience: single-sample probability."""
        with torch.no_grad():
            logit = self.forward(x)
            return float(torch.sigmoid(logit)[0, 0])
