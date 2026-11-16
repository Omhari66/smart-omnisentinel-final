"""
models/audio_cnn.py
-------------------
Day 8 Deliverable: 1D-CNN Audio Encoder & Multimodal Fusion Layer.

Architecture:
  Conv1d(13, 32, k=5) -> BN -> ReLU -> MaxPool
  Conv1d(32, 64, k=3) -> BN -> ReLU -> MaxPool
  Conv1d(64, 32, k=3) -> BN -> ReLU -> GlobalAvgPool -> 32-dim embedding

Fusion:
  concat(visual_gru_output[256], audio_embedding[32]) -> Linear(288 -> 1)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Audio1DCNN(nn.Module):
    """
    1D-CNN encoder for MFCC acoustic spectrograms.
    Input:  (batch, in_channels=13, time_frames=100)
    Output: (batch, embed_dim=32)
    """

    def __init__(self, in_channels: int = 13, embed_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # 100 -> 50
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # 50 -> 25
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),      # Global average pooling
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        return self.dropout(h.squeeze(-1))


class MultimodalVisualAudioFusion(nn.Module):
    """
    Fuses 256-dim Visual GRU embedding with 32-dim Audio embedding.
    """

    def __init__(self, visual_dim: int = 256, audio_dim: int = 32, num_classes: int = 1):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.fusion_linear = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, visual_emb: torch.Tensor, audio_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = visual_emb.shape[0]
        if audio_emb is None:
            # Silent video support: zero vector, graceful execution
            audio_emb = torch.zeros((b, self.audio_dim), dtype=visual_emb.dtype, device=visual_emb.device)

        fused = torch.cat([visual_emb, audio_emb], dim=-1)
        out = self.fusion_linear(fused)
        return torch.sigmoid(out) if out.shape[-1] == 1 else out
