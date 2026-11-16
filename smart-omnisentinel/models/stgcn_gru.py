"""
models/stgcn_gru.py
-------------------
Day 4 Deliverable: Hybrid ST-GCN + GRU Architecture.

Architecture:
  Skeleton Graph (17 joints x 3 coords) ->
  ST-GCN Spatial Encoder -> 128-dim per frame ->
  Concatenate 6-dim Inter-Person Proximity (128 + 6 = 134) ->
  2-Layer Temporal GRU (hidden_size=256, dropout=0.3) ->
  Linear(256 -> 1) + Sigmoid (or Linear(256 -> 2) Cross-Entropy)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ml.models.st_gcn import COCO_BONES, NUM_COCO_JOINTS
from models.stgcn import STGCNBackbone


class SpatialGraphConvFast(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 3, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        b, c3, t, v = h.shape
        c = c3 // 3
        h = h.view(b, 3, c, t, v)
        return torch.einsum('bkc tv, kvw -> bctw', h, A)


class STGCNBlockFast(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.gconv = SpatialGraphConvFast(in_c, out_c)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.gconv(x, A)))


class VelocitySTGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolution + GRU with 6 input channels (x, y, conf, dx, dy, dconf).
    Trained directly on RWF-2000 surveillance skeletons.
    """
    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()
        A = np.zeros((3, 17, 17), dtype=np.float32)
        A[0] = np.eye(17)
        for i, j in COCO_BONES:
            A[1, i, j] = 1.0; A[1, j, i] = 1.0
            A[2, i, j] = 1.0; A[2, j, i] = 1.0
        for k in range(3):
            deg = np.sum(A[k], axis=1) + 1e-5
            A[k] = A[k] / deg[:, None]
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

        self.block1 = STGCNBlockFast(in_channels, 64)
        self.block2 = STGCNBlockFast(64, 128)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, self.A)
        h = self.block2(h, self.A)
        frame_emb = torch.mean(h, dim=-1).permute(0, 2, 1)
        gru_out, _ = self.gru(frame_emb)
        return self.fc(gru_out[:, -1, :])


class STGCNGRUClassifier(nn.Module):
    """
    Combined Spatio-Temporal Graph Convolution + Gated Recurrent Unit classifier.
    """

    def __init__(
        self,
        in_channels: int = 3,
        proximity_dim: int = 6,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stgcn = STGCNBackbone(in_channels=in_channels, dropout=dropout)

        gru_input_dim = 128 + proximity_dim
        self.proximity_dim = proximity_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        proximity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, channels=3, time=16, joints=17) or (batch, time, 51)
            proximity: Optional (batch, time=16, proximity_dim=6) or (batch, proximity_dim=6)
        Returns:
            logits: (batch, num_classes)
        """
        # 1. Spatial Graph feature extraction -> (batch, time=16, 128)
        frame_embs = self.stgcn(x)
        b, t, _ = frame_embs.shape

        # 2. Inter-person proximity concatenation
        if self.proximity_dim > 0:
            if proximity is None:
                proximity = torch.zeros((b, t, self.proximity_dim), dtype=x.dtype, device=x.device)
            elif proximity.dim() == 2:
                # Repeat across temporal sequence
                proximity = proximity.unsqueeze(1).expand(-1, t, -1)

            gru_input = torch.cat([frame_embs, proximity], dim=-1)
        else:
            gru_input = frame_embs

        # 3. Temporal GRU
        gru_out, _ = self.gru(gru_input)
        last_step = gru_out[:, -1, :]

        # 4. Final classification
        logits = self.classifier(last_step)
        return logits
