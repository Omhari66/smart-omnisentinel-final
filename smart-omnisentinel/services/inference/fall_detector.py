"""
services/inference/fall_detector.py
-------------------------------------
Hybrid fall detector: geometric rules + optional binary classifier.

Why hybrid:
- Geometric rules give high recall (catch most falls, few misses)
- Classifier reduces false positives from sitting/bending
- Both must agree for a fall detection to be emitted

Geometric trigger conditions:
1. Head keypoint Y drops significantly relative to hip keypoint Y within N frames
2. Body bounding box aspect ratio crosses threshold (vertical → horizontal)
3. Person was previously standing (body taller than wide by 1.5x)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from core.logger import get_logger
from services.inference.pose_estimator import PoseEstimate

logger = get_logger(__name__)

# Geometric thresholds
ASPECT_RATIO_STANDING_MIN = 1.4   # height/width ratio when standing
ASPECT_RATIO_FALLEN_MAX = 1.0     # height/width ratio when fallen
HEAD_DROP_FRAMES = 8              # Frames within which head must drop
HEAD_DROP_FRACTION = 0.25         # Head must drop >25% of body height
STANDING_HISTORY_FRAMES = 10     # Person must have been standing for this long


@dataclass
class FallState:
    """Per-track fall detection state."""
    track_id: int
    aspect_ratio_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    head_y_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=HEAD_DROP_FRAMES + 5)
    )  # (timestamp, normalized_y)
    was_standing: bool = False
    fall_triggered: bool = False
    trigger_timestamp: Optional[float] = None


@dataclass
class FallDetectionResult:
    track_id: int
    geometric_triggered: bool
    classifier_confidence: float  # 0.0 if classifier not loaded
    is_fall: bool                 # True only if both agree (or geometric + no classifier)
    confidence: float             # Final reported confidence


class FallDetector:
    """
    Detects person falls using geometric pose rules with optional classifier confirmation.
    Maintains per-track state to detect the transition from standing to fallen.
    """

    def __init__(self):
        self._states: Dict[int, FallState] = {}

    def _get_state(self, track_id: int) -> FallState:
        if track_id not in self._states:
            self._states[track_id] = FallState(track_id=track_id)
        return self._states[track_id]

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def detect(self, pose: PoseEstimate) -> FallDetectionResult:
        """
        Evaluate fall likelihood for a single tracked person.
        Returns FallDetectionResult with is_fall flag.
        """
        state = self._get_state(pose.track_id)

        # --- Compute current aspect ratio ---
        x1, y1, x2, y2 = pose.bbox_xyxy
        width = x2 - x1
        height = y2 - y1
        if width <= 0:
            return FallDetectionResult(
                track_id=pose.track_id,
                geometric_triggered=False,
                classifier_confidence=0.0,
                is_fall=False,
                confidence=0.0,
            )
        aspect_ratio = height / width
        state.aspect_ratio_history.append(aspect_ratio)

        # --- Track standing history ---
        if aspect_ratio >= ASPECT_RATIO_STANDING_MIN:
            state.was_standing = True

        # --- Track head Y position ---
        head_kp = pose.get_keypoint("nose")
        if head_kp is not None:
            state.head_y_history.append((time.time(), head_kp[1]))

        # --- Geometric trigger check ---
        geometric_triggered = False

        if state.was_standing and aspect_ratio <= ASPECT_RATIO_FALLEN_MAX:
            # Check if aspect ratio transitioned from standing to fallen recently
            if len(state.aspect_ratio_history) >= 5:
                recent_max = max(list(state.aspect_ratio_history)[-8:])
                if recent_max >= ASPECT_RATIO_STANDING_MIN:
                    geometric_triggered = True

        # --- Head drop check (secondary trigger) ---
        if (
            not geometric_triggered
            and len(state.head_y_history) >= HEAD_DROP_FRAMES
            and state.was_standing
        ):
            oldest_y = state.head_y_history[0][1]
            current_y = state.head_y_history[-1][1]
            # In normalized coords, Y increases downward
            drop = current_y - oldest_y
            body_height_norm = (y2 - y1)  # already in pixels; normalize to frame
            if drop > HEAD_DROP_FRACTION:
                geometric_triggered = True

        if not geometric_triggered:
            return FallDetectionResult(
                track_id=pose.track_id,
                geometric_triggered=False,
                classifier_confidence=0.0,
                is_fall=False,
                confidence=0.0,
            )

        # --- Geometric triggered: optional classifier confirmation ---
        classifier_conf = self._run_classifier(pose)

        # Determine final decision
        if classifier_conf > 0.0:
            # Classifier loaded: both must agree
            is_fall = classifier_conf >= 0.60
            confidence = classifier_conf
        else:
            # No classifier: trust geometric rules with moderate confidence
            is_fall = True
            confidence = 0.72

        if is_fall and not state.fall_triggered:
            state.fall_triggered = True
            state.trigger_timestamp = time.time()
            logger.info(
                "fall_detected",
                track_id=pose.track_id,
                geometric=geometric_triggered,
                classifier_conf=classifier_conf,
                confidence=confidence,
            )

        return FallDetectionResult(
            track_id=pose.track_id,
            geometric_triggered=geometric_triggered,
            classifier_confidence=classifier_conf,
            is_fall=is_fall,
            confidence=confidence,
        )

    def _run_classifier(self, pose: PoseEstimate) -> float:
        """
        Optional binary classifier pass.
        Returns 0.0 if model not loaded (geometric rule used alone).
        """
        from services.inference.model_loader import get_models
        models = get_models()

        if models.fall_classifier is None or not pose.is_valid:
            return 0.0

        try:
            import torch
            # Flatten keypoints + confidences as feature vector
            features = np.concatenate([
                pose.keypoints.flatten(),
                pose.keypoint_conf,
            ])
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logit = models.fall_classifier(x)
                prob = float(torch.sigmoid(logit)[0, 0])
            return prob
        except Exception as exc:
            logger.warning("fall_classifier_error", error=str(exc))
            return 0.0
