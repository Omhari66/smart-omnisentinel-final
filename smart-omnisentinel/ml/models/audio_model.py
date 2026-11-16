"""
ml/models/audio_model.py
------------------------
Multimodal Audio Branch & Cross-Modal Late Fusion for Violence Detection.

Acoustic path:
  1-second sliding audio window -> 13/39-dim MFCC spectrogram ->
  1D-CNN encoder -> 32-dim audio latent embedding.

Cross-Modal Late Fusion:
  Fuses Visual ST-GCN embedding (128-dim) + Proximity latent (16-dim) +
  Audio latent (32-dim) with an adaptive gating mechanism.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MFCC_CHANNELS = 13  # 13 MFCC coefficients (or 39 with delta + delta-delta)
AUDIO_EMBED_DIM = 32
VISUAL_EMBED_DIM = 128
PROXIMITY_EMBED_DIM = 16


class Audio1DCNN(nn.Module):
    """
    Lightweight 1D-CNN acoustic encoder for violence audio signals
    (e.g., shouts, screams, blunt impacts, breaking glass).

    Input:  (batch, in_channels=13, time_frames=32)
    Output: (batch, audio_embed_dim=32)
    """

    def __init__(
        self,
        in_channels: int = MFCC_CHANNELS,
        embed_dim: int = AUDIO_EMBED_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # 32 -> 16
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # 16 -> 8
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),      # Global temporal pooling
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time_frames)
        Returns:
            latent: (batch, embed_dim)
        """
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.dropout(h.squeeze(-1))
        return h


class MultimodalFusionClassifier(nn.Module):
    """
    Adaptive Cross-Modal Late Fusion layer.

    Merges visual features (ST-GCN), kinematic proximity, and audio cues.
    An adaptive sigmoid gating gate dynamically weights visual vs. acoustic
    confidence depending on noise levels or lighting degradation.
    """

    def __init__(
        self,
        visual_dim: int = VISUAL_EMBED_DIM,
        proximity_dim: int = PROXIMITY_EMBED_DIM,
        audio_dim: int = AUDIO_EMBED_DIM,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.proximity_dim = proximity_dim
        self.audio_dim = audio_dim

        # Gating network: computes acoustic relevance alpha in [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        total_features = visual_dim + proximity_dim + audio_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        visual_latent: torch.Tensor,
        proximity_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_latent: (batch, visual_dim)
            proximity_latent: (batch, proximity_dim)
            audio_latent: Optional (batch, audio_dim)
        Returns:
            logits: (batch, num_classes)
            gate_weight: (batch, 1) indicating audio contribution
        """
        batch_size = visual_latent.shape[0]

        if audio_latent is None:
            audio_latent = torch.zeros(
                (batch_size, self.audio_dim),
                dtype=visual_latent.dtype,
                device=visual_latent.device,
            )

        # Adaptive gating between audio and visual signals
        gate = self.gate_net(torch.cat([visual_latent, audio_latent], dim=-1))

        # Modulate audio features with the learned gate
        gated_audio = gate * audio_latent

        fused = torch.cat([visual_latent, proximity_latent, gated_audio], dim=-1)
        logits = self.classifier(fused)
        return logits, gate
