"""
models/weapon_detector.py
-------------------------
Day 12 Deliverable: Secondary Signal Weapon Detector.

Detects edged/impact weapons from COCO classes:
  - knife (class 43)
  - baseball bat (class 34)
  - bottle (class 39)
  - scissors (class 76)

Inference Rule:
  If a weapon is detected within 0.5m of an active track,
  the threat risk score is multiplied by 1.5x with weapon metadata tagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO weapon-like object class IDs
WEAPON_CLASS_MAP = {
    34: "baseball_bat",
    39: "bottle",
    43: "knife",
    76: "scissors",
}


@dataclass
class WeaponDetection:
    weapon_type: str
    confidence: float
    bbox_xyxy: List[float]
    centroid: Tuple[float, float]
    associated_track_id: Optional[int] = None


class WeaponDetector:
    """
    Lightweight secondary signal weapon detector.
    """

    def __init__(self, confidence_threshold: float = 0.35):
        self.conf_thresh = confidence_threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8n.pt")
            except Exception:
                self._model = None
        return self._model

    def detect_weapons(
        self,
        frame: np.ndarray,
        tracks: List[object],
        max_proximity_px: float = 120.0,
    ) -> List[WeaponDetection]:
        """
        Detects weapons and associates them with nearby tracked individuals.
        """
        model = self._get_model()
        if model is None:
            return []

        results = model.predict(frame, classes=list(WEAPON_CLASS_MAP.keys()), conf=self.conf_thresh, verbose=False)
        detections: List[WeaponDetection] = []

        if not results or len(results[0].boxes) == 0:
            return []

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0

            # Associate with closest person track
            closest_tid = None
            min_dist = max_proximity_px
            for t in tracks:
                tc = getattr(t, "centroid", (0, 0))
                d = math.hypot(cx - tc[0], cy - tc[1])
                if d < min_dist:
                    min_dist = d
                    closest_tid = getattr(t, "track_id", None)

            detections.append(
                WeaponDetection(
                    weapon_type=WEAPON_CLASS_MAP.get(cls_id, "unknown_weapon"),
                    confidence=conf,
                    bbox_xyxy=xyxy,
                    centroid=(cx, cy),
                    associated_track_id=closest_tid,
                )
            )

        return detections
