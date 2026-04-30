"""
ml/models/temporal_model.py
-----------------------------
CNN + GRU temporal classifier for violence / distress detection.

Architecture:
  Input: (batch, WINDOW_SIZE=16, FEATURE_DIM=55)
  → Linear projection (55 → 128)  [acts as lightweight "CNN" feature extractor]
  → ReLU + Dropout(0.3)
  → GRU(128 → 256, num_layers=2, dropout=0.3)
  → Take last hidden state
  → FC(256 → 128) → ReLU → Dropout(0.3)
  → FC(128 → 3)  [normal, violent_interaction, person_distress]
  → LogSoftmax (use NLLLoss during training, Softmax at inference)

Design rationale:
- Linear projection replaces a per-frame CNN since input features are
  already extracted (pose keypoints + motion) — no raw pixel input
- GRU chosen over LSTM: fewer parameters, similar performance on short sequences
- 2-layer GRU captures both short-term motion and longer behavioral context
- Dropout applied aggressively to combat small training set overfitting
"""

from __future__ import annotations

import torch
import torch.nn as nn

WINDOW_SIZE = 16
FEATURE_DIM = 34
NUM_CLASSES = 2


class TemporalClassifier(nn.Module):
    """
    CNN+GRU temporal behavior classifier.
    Input: (batch, WINDOW_SIZE, FEATURE_DIM)
    Output: (batch, NUM_CLASSES) logits
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Frame-level feature projection (replaces per-frame CNN)
        self.frame_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        # Temporal GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,         # Input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,      # Causal: no look-ahead for real-time use
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, WINDOW_SIZE, FEATURE_DIM)
        Returns:
            logits: Tensor of shape (batch, NUM_CLASSES)
        """
        # Encode each frame independently
        batch, seq, feat = x.shape
        x_flat = x.view(batch * seq, feat)
        encoded = self.frame_encoder(x_flat)
        encoded = encoded.view(batch, seq, -1)

        # GRU over the sequence
        gru_out, _ = self.gru(encoded)

        # Use last time step output
        last_hidden = gru_out[:, -1, :]

        # Classify
        logits = self.classifier(last_hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


class FallBinaryClassifier(nn.Module):
    """
    Lightweight binary classifier for fall detection.
    Input: (batch, 51) = flattened 17 keypoints (x,y) + 17 confidences
    Output: (batch, 1) logit (sigmoid → probability of fall)
    """

    def __init__(self, input_dim: int = 51, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
