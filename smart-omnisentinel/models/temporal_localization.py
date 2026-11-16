"""
models/temporal_localization.py
-------------------------------
Day 10 Deliverable: Multi-Task Temporal Action Localization Model.

Outputs:
  1. Classification Logits: (batch, 2) [Normal vs Violent]
  2. Action Temporal Boundaries: (batch, 2) [start_frac, end_frac] in [0.0, 1.0]

Loss Objective:
  L = L_cls + 0.5 * L_boundary
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.stgcn_gru import STGCNGRUClassifier


class TemporalLocalizationHead(nn.Module):
    """
    Dual-head network: Classification + Temporal Boundary Regression.
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Boundary localization regression head [start_boundary, end_boundary]
        self.boundary_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # Fraction of temporal window in [0.0, 1.0]
        )

    def forward(self, gru_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_features: (batch, hidden_dim=256)
        Returns:
            cls_logits: (batch, num_classes)
            boundaries: (batch, 2) - [start_frac, end_frac]
        """
        logits = self.cls_head(gru_features)
        boundaries = self.boundary_head(gru_features)
        # Ensure start <= end by sorting
        start = torch.min(boundaries[:, 0], boundaries[:, 1]).unsqueeze(-1)
        end = torch.max(boundaries[:, 0], boundaries[:, 1]).unsqueeze(-1)
        ordered_boundaries = torch.cat([start, end], dim=-1)
        return logits, ordered_boundaries


def format_incident_timestamps(
    window_start_sec: float,
    window_duration_sec: float,
    start_frac: float,
    end_frac: float,
    confidence: float,
) -> Dict[str, str | float]:
    """
    Formats detected incident into user-facing timestamp range.
    Example: 'Violence detected: 00:03.2 – 00:05.8'
    """
    incident_start = window_start_sec + (start_frac * window_duration_sec)
    incident_end = window_start_sec + (end_frac * window_duration_sec)

    def _sec_to_mmss(s: float) -> str:
        mm = int(s // 60)
        ss = s % 60
        return f"{mm:02d}:{ss:04.1f}"

    return {
        "start_time_str": _sec_to_mmss(incident_start),
        "end_time_str": _sec_to_mmss(incident_end),
        "start_seconds": round(incident_start, 2),
        "end_seconds": round(incident_end, 2),
        "duration_seconds": round(incident_end - incident_start, 2),
        "confidence": round(confidence, 3),
        "display_banner": f"Violence detected: {_sec_to_mmss(incident_start)} – {_sec_to_mmss(incident_end)}",
    }

