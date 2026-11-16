"""
ml/models/st_gcn.py
-------------------
Spatio-Temporal Graph Convolutional Network (ST-GCN) for Skeleton-Based Action
and Violence Recognition.

Designed for 17 COCO keypoints:
  Input shape: (batch_size, in_channels=3, temporal_len=16, num_nodes=17)
  where in_channels = (x, y, confidence)

Joint Topology (COCO 17 Keypoints):
  0: Nose
  1: Left Eye,     2: Right Eye
  3: Left Ear,     4: Right Ear
  5: Left Shoulder, 6: Right Shoulder
  7: Left Elbow,   8: Right Elbow
  9: Left Wrist,  10: Right Wrist
 11: Left Hip,    12: Right Hip
 13: Left Knee,   14: Right Knee
 15: Left Ankle,  16: Right Ankle
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_COCO_JOINTS = 17

# Anatomical skeleton edges (kinematic bones)
COCO_BONES: List[Tuple[int, int]] = [
    # Head connections
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body connections
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso connections
    (5, 11), (6, 12), (11, 12),
    # Lower body connections
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class SkeletonGraph:
    """
    Constructs the spatial adjacency matrix for 17 COCO human joints.
    Implements the spatial configuration partitioning strategy:
      Subset 0: Self-loop (root node)
      Subset 1: Inward edges (closer to body center of gravity: hip/torso)
      Subset 2: Outward edges (further from body center of gravity)
    """

    def __init__(self, num_nodes: int = NUM_COCO_JOINTS, max_hop: int = 1):
        self.num_nodes = num_nodes
        self.max_hop = max_hop
        self.A = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self) -> np.ndarray:
        # Distance matrix initialization
        dist_matrix = np.full((self.num_nodes, self.num_nodes), fill_value=np.inf)
        np.fill_diagonal(dist_matrix, 0)

        for i, j in COCO_BONES:
            dist_matrix[i, j] = 1
            dist_matrix[j, i] = 1

        # Compute shortest hop distances
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

        # Build 3-subset spatial configuration adjacency
        A = np.zeros((3, self.num_nodes, self.num_nodes), dtype=np.float32)

        # Body center defined as midpoint of hips (11, 12) and shoulders (5, 6)
        center_distance = [
            min(dist_matrix[i, 5], dist_matrix[i, 6], dist_matrix[i, 11], dist_matrix[i, 12])
            for i in range(self.num_nodes)
        ]

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if dist_matrix[i, j] == 0:
                    A[0, i, j] = 1.0
                elif dist_matrix[i, j] <= self.max_hop:
                    if center_distance[j] < center_distance[i]:
                        A[1, i, j] = 1.0  # Inward
                    else:
                        A[2, i, j] = 1.0  # Outward

        # Normalize each partition matrix: D^(-1) * A
        for p in range(3):
            d = np.sum(A[p], axis=1)
            d[d == 0] = 1.0
            A[p] = A[p] / d[:, None]

        return A


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer over the skeleton graph partitions.
    """

    def __init__(self, in_channels: int, out_channels: int, num_partitions: int = 3):
        super().__init__()
        self.num_partitions = num_partitions
        self.conv = nn.Conv2d(
            in_channels=in_channels * num_partitions,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time, num_nodes)
            A: (num_partitions, num_nodes, num_nodes)
        Returns:
            out: (batch, out_channels, time, num_nodes)
        """
        batch, c, t, v = x.shape
        partitions = []

        for p in range(self.num_partitions):
            x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * t * c, v)
            a_p = A[p].to(x.device, dtype=x.dtype)
            z = torch.matmul(x_reshaped, a_p)
            z = z.view(batch, t, c, v).permute(0, 2, 1, 3)
            partitions.append(z)

        x_concat = torch.cat(partitions, dim=1)
        out = self.conv(x_concat)
        return out


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolution Block:
      Spatial Graph Conv -> BatchNorm -> ReLU -> Dropout ->
      Temporal 1D Conv (kernel 9x1) -> BatchNorm -> Residual Add -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.sgcn = SpatialGraphConv(in_channels, out_channels)
        self.bn_spatial = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        padding = (temporal_kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.sgcn(x, A)
        x = self.bn_spatial(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.tcn(x)
        x = self.relu(x + res)
        return x


class STGCNClassifier(nn.Module):
    """
    End-to-End Spatio-Temporal Graph Convolutional Network for Violence Detection.

    Args:
        in_channels: 3 by default (x, y, confidence)
        num_classes: 2 (0: Normal, 1: Violent Interaction)
        proximity_dim: Additional scalar features (e.g. inter-person distance, relative speed)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        proximity_dim: int = 8,
        temporal_len: int = 16,
        num_nodes: int = NUM_COCO_JOINTS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.graph = SkeletonGraph(num_nodes=num_nodes)
        self.register_buffer("A", torch.tensor(self.graph.A, dtype=torch.float32))

        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # ST-GCN Backbone layers
        self.block1 = STGCNBlock(in_channels, 64, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64, 64, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64, 128, stride=2, dropout=dropout)
        self.block4 = STGCNBlock(128, 128, stride=1, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proximity_dim = proximity_dim
        if proximity_dim > 0:
            self.proximity_encoder = nn.Sequential(
                nn.Linear(proximity_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
            )
            classifier_input_dim = 128 + 16
        else:
            classifier_input_dim = 128

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        proximity_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, in_channels, time, num_nodes)
               or (batch, time, num_nodes * in_channels)
            proximity_features: Optional Tensor of shape (batch, proximity_dim)
        Returns:
            logits: (batch, num_classes)
        """
        if x.dim() == 3:
            b, t, c_v = x.shape
            v = NUM_COCO_JOINTS
            c = c_v // v
            x = x.view(b, t, v, c).permute(0, 3, 1, 2)

        b, c, t, v = x.shape

        x_flat = x.permute(0, 1, 3, 2).reshape(b, c * v, t)
        x_norm = self.data_bn(x_flat)
        x = x_norm.view(b, c, v, t).permute(0, 1, 3, 2)

        h = self.block1(x, self.A)
        h = self.block2(h, self.A)
        h = self.block3(h, self.A)
        h = self.block4(h, self.A)

        visual_latent = self.pool(h).view(b, -1)

        if self.proximity_dim > 0:
            if proximity_features is None:
                proximity_features = torch.zeros(
                    (b, self.proximity_dim), dtype=x.dtype, device=x.device
                )
            prox_latent = self.proximity_encoder(proximity_features)
            combined = torch.cat([visual_latent, prox_latent], dim=-1)
        else:
            combined = visual_latent

        logits = self.classifier(combined)
        return logits

    def extract_joint_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hook for Grad-CAM / Attention visualization.
        """
        if x.dim() == 3:
            b, t, c_v = x.shape
            v = NUM_COCO_JOINTS
            c = c_v // v
            x = x.view(b, t, v, c).permute(0, 3, 1, 2)

        b, c, t, v = x.shape
        x_flat = x.permute(0, 1, 3, 2).reshape(b, c * v, t)
        x_norm = self.data_bn(x_flat)
        x = x_norm.view(b, c, v, t).permute(0, 1, 3, 2)

        h = self.block1(x, self.A)
        h = self.block2(h, self.A)
        h = self.block3(h, self.A)
        h = self.block4(h, self.A)
        return h
