"""
services/event_analysis/confidence_smoother.py
------------------------------------------------
Exponential Moving Average smoother for per-camera per-event-type confidence.
Prevents single-frame confidence spikes from triggering events.

Usage:
    smoother = ConfidenceSmoother(alpha=0.35)
    smoothed = smoother.update("cam1", "VIOLENT_INTERACTION", raw_conf=0.87)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple


class ConfidenceSmoother:
    """
    EMA smoother: smoothed_t = α × raw_t + (1-α) × smoothed_{t-1}
    State is maintained per (camera_id, event_type) pair.
    """

    def __init__(self, alpha: float = 0.35):
        assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
        self.alpha = alpha
        # (camera_id, event_type) → smoothed confidence
        self._state: Dict[Tuple[str, str], float] = defaultdict(float)

    def update(self, camera_id: str, event_type: str, raw_confidence: float) -> float:
        key = (camera_id, event_type)
        prev = self._state[key]
        smoothed = self.alpha * raw_confidence + (1.0 - self.alpha) * prev
        self._state[key] = smoothed
        return smoothed

    def reset(self, camera_id: str, event_type: str) -> None:
        """Reset state after event is confirmed or resolved."""
        key = (camera_id, event_type)
        self._state.pop(key, None)

    def get(self, camera_id: str, event_type: str) -> float:
        return self._state.get((camera_id, event_type), 0.0)
