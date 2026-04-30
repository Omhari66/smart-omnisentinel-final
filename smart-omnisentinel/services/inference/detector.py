"""
services/inference/detector.py
-------------------------------
YOLOv8n person detection wrapper.
Runs detection every N frames (configurable skip).
Returns normalized bounding boxes and confidence scores.
Only class 0 (person) detections are returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

PERSON_CLASS_ID = 0
MIN_DETECTION_CONFIDENCE = 0.40   # Below this, ignore regardless of settings
MIN_BBOX_AREA_FRACTION = 0.001    # Ignore tiny detections (< 0.1% of frame)


@dataclass
class Detection:
    """A single person detection result."""
    track_id: Optional[int]        # Assigned by tracker; None until tracked
    bbox_xyxy: List[float]         # [x1, y1, x2, y2] in pixel coords
    confidence: float              # Detector confidence 0.0–1.0
    frame_number: int
    timestamp: float               # Unix timestamp of this frame


class PersonDetector:
    """
    Wraps the YOLO detector for person-only detection.
    Runs inference only every `skip_frames` frames to save compute.
    Caches the last result for skipped frames.
    """

    def __init__(self):
        settings = get_settings()
        self._device = settings.inference.device
        self._skip_n = settings.inference.detection_skip_frames
        self._frame_counter = 0
        self._last_raw_result = None

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> List[Detection]:
        """
        Run person detection on a frame.
        Returns list of Detection objects (empty if no persons found).
        On skipped frames, returns detections from the last inference pass.
        """
        from services.inference.model_loader import get_models
        models = get_models()

        self._frame_counter += 1
        run_inference = (self._frame_counter % self._skip_n == 0)

        if run_inference or self._last_raw_result is None:
            if models.detector is None:
                return []
            try:
                results = models.detector.predict(
                    frame,
                    classes=[PERSON_CLASS_ID],
                    conf=MIN_DETECTION_CONFIDENCE,
                    verbose=False,
                    device=self._device,
                )
                self._last_raw_result = results[0] if results else None
            except Exception as exc:
                logger.error("detection_failed", error=str(exc))
                return []

        if self._last_raw_result is None:
            return []

        detections = self._parse_result(
            self._last_raw_result, frame.shape, frame_number, timestamp
        )
        return detections

    def _parse_result(
        self,
        result,
        frame_shape: tuple,
        frame_number: int,
        timestamp: float,
    ) -> List[Detection]:
        h, w = frame_shape[:2]
        frame_area = h * w
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < MIN_DETECTION_CONFIDENCE:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area / frame_area < MIN_BBOX_AREA_FRACTION:
                continue  # Skip tiny detections (distant persons, noise)

            detections.append(Detection(
                track_id=None,
                bbox_xyxy=[x1, y1, x2, y2],
                confidence=conf,
                frame_number=frame_number,
                timestamp=timestamp,
            ))

        return detections
