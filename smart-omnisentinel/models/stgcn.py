"""
models/stgcn.py
---------------
Day 3 Deliverable: Spatio-Temporal Graph Convolutional Network backbone.
COCO 17 Joints, 30 Kinematic Bone Edges, Spatial Configuration Partitioning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.st_gcn import COCO_BONES, NUM_COCO_JOINTS, SkeletonGraph, SpatialGraphConv, STGCNBlock


class STGCNBackbone(nn.Module):
    """
    ST-GCN graph encoder:
    Input:  (batch, in_channels=3, time=16, joints=17)
    Output: (batch, time=16, 128) - frame-level graph embeddings for GRU
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.graph = SkeletonGraph(num_nodes=NUM_COCO_JOINTS)
        self.register_buffer("A", torch.tensor(self.graph.A, dtype=torch.float32))

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_COCO_JOINTS)

        self.block1 = STGCNBlock(in_channels, 64, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64, 64, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64, 128, stride=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            frame_embeddings: (batch, time, 128)
        """
        # x shape: (batch, channels=3, time=16, joints=17)
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
        h = self.block3(h, self.A)  # (batch, 128, time, joints)

        # Average pool across joints (spatial pooling)
        # Keeps time dimension intact for GRU
        frame_emb = torch.mean(h, dim=-1).permute(0, 2, 1)  # (batch, time, 128)
        return frame_emb
