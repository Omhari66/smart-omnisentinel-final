"""
losses/boundary_loss.py
-----------------------
Day 10 Deliverable: Temporal Action Boundary Loss & Multi-Task Objective.

Loss Formulation:
  L_total = L_cls + λ * L_boundary
where:
  L_boundary = |pred_start - gt_start| + |pred_end - gt_end|
  λ = 0.5
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBoundaryLoss(nn.Module):
    """
    Computes boundary regression loss combined with classification cross-entropy.
    """

    def __init__(self, lambda_boundary: float = 0.5):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.cls_criterion = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(
        self,
        cls_logits: torch.Tensor,
        boundary_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        boundary_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_logits: (batch, 2)
            boundary_preds: (batch, 2) - [start_frac, end_frac] in [0, 1]
            cls_targets: (batch,) - class index (0 or 1)
            boundary_targets: (batch, 2) - ground truth [start_frac, end_frac]
        """
        loss_cls = self.cls_criterion(cls_logits, cls_targets)

        # Boundary loss is computed only on positive (violent) instances
        pos_mask = (cls_targets == 1).float()
        if pos_mask.sum() > 0:
            l1_diff = self.l1_loss(boundary_preds, boundary_targets).sum(dim=-1)
            loss_boundary = (l1_diff * pos_mask).sum() / pos_mask.sum()
        else:
            loss_boundary = torch.tensor(0.0, device=cls_logits.device)

        total_loss = loss_cls + self.lambda_boundary * loss_boundary
        return total_loss
