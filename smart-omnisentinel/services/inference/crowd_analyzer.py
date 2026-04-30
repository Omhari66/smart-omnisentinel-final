"""
services/inference/crowd_analyzer.py
--------------------------------------
Crowd aggression / panic detector.

V1 (MVP): Heuristic-based using optical flow + person density.
V2: Full LSTM on aggregate motion feature sequences.

Detects:
  - Sudden crowd dispersal (panic/explosion response pattern)
  - High-density pushing / surging
  - Rapid direction reversal in crowd movement

The InferenceRunner calls this module's analyze() method
after computing optical flow.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)

# Thresholds for heuristic triggers
MIN_PEOPLE_FOR_CROWD = 4
HIGH_FLOW_MAGNITUDE = 4.0        # Optical flow magnitude threshold
DISPERSAL_FLOW_CHANGE = 2.5      # Sudden jump in flow magnitude
DIRECTION_VARIANCE_THRESHOLD = 1.8  # High variance = many directions = chaos
HISTORY_FRAMES = 16


@dataclass
class CrowdFeatureFrame:
    """Aggregate scene features for one frame."""
    timestamp: float
    people_count: int
    flow_magnitude: float
    flow_direction_variance: float
    centroid_spread: float          # How spread out the crowd is


@dataclass
class CrowdAnalysisResult:
    detected: bool
    confidence: float
    reason: str
    people_count: int


class CrowdAnalyzer:
    """
    Heuristic crowd aggression detector.
    Maintains a rolling window of crowd feature frames.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._history: Deque[CrowdFeatureFrame] = deque(maxlen=HISTORY_FRAMES)

    def analyze(
        self,
        people_count: int,
        flow_magnitude: float,
        flow_vectors: Optional[np.ndarray] = None,   # (H, W, 2) optical flow field
        track_centroids: Optional[List[Tuple[float, float]]] = None,
    ) -> CrowdAnalysisResult:
        """
        Analyze current frame for crowd aggression signals.

        Args:
            people_count: Number of detected persons this frame.
            flow_magnitude: Mean optical flow magnitude.
            flow_vectors: Full optical flow field (for direction variance).
            track_centroids: List of (x, y) person centroids.
        """
        if people_count < MIN_PEOPLE_FOR_CROWD:
            return CrowdAnalysisResult(
                detected=False, confidence=0.0,
                reason="insufficient_people", people_count=people_count
            )

        # Compute direction variance from flow field
        direction_variance = 0.0
        if flow_vectors is not None:
            angles = np.arctan2(flow_vectors[..., 1], flow_vectors[..., 0])
            # Circular variance
            sin_mean = np.mean(np.sin(angles))
            cos_mean = np.mean(np.cos(angles))
            direction_variance = float(1.0 - np.sqrt(sin_mean**2 + cos_mean**2))

        # Compute centroid spread (std dev of positions)
        centroid_spread = 0.0
        if track_centroids and len(track_centroids) >= 2:
            xs = [c[0] for c in track_centroids]
            ys = [c[1] for c in track_centroids]
            centroid_spread = float(np.std(xs) + np.std(ys))

        frame = CrowdFeatureFrame(
            timestamp=time.time(),
            people_count=people_count,
            flow_magnitude=flow_magnitude,
            flow_direction_variance=direction_variance,
            centroid_spread=centroid_spread,
        )
        self._history.append(frame)

        return self._evaluate()

    def _evaluate(self) -> CrowdAnalysisResult:
        """Score the current history window for aggression patterns."""
        if len(self._history) < 4:
            return CrowdAnalysisResult(
                detected=False, confidence=0.0,
                reason="insufficient_history",
                people_count=self._history[-1].people_count if self._history else 0
            )

        recent = list(self._history)[-8:]
        latest = recent[-1]
        confidence = 0.0
        reasons = []

        # Signal 1: High sustained flow magnitude with crowd
        if latest.flow_magnitude >= HIGH_FLOW_MAGNITUDE:
            flow_factor = min(latest.flow_magnitude / HIGH_FLOW_MAGNITUDE, 2.0)
            confidence += 0.25 * flow_factor
            reasons.append("high_flow")

        # Signal 2: Sudden flow magnitude spike (panic dispersal)
        if len(recent) >= 4:
            prev_magnitudes = [f.flow_magnitude for f in recent[:-2]]
            avg_prev = sum(prev_magnitudes) / len(prev_magnitudes)
            if latest.flow_magnitude - avg_prev > DISPERSAL_FLOW_CHANGE:
                confidence += 0.30
                reasons.append("flow_spike")

        # Signal 3: High direction variance (crowd moving in all directions)
        if latest.flow_direction_variance > DIRECTION_VARIANCE_THRESHOLD:
            confidence += 0.25
            reasons.append("direction_chaos")

        # Signal 4: People count factor
        people_factor = min(latest.people_count / 8.0, 1.0)
        confidence += 0.20 * people_factor

        # Clamp
        confidence = min(confidence, 1.0)
        detected = confidence >= 0.55

        return CrowdAnalysisResult(
            detected=detected,
            confidence=round(confidence, 3),
            reason=",".join(reasons) if reasons else "none",
            people_count=latest.people_count,
        )
