"""
services/inference/tracker.py
------------------------------
ByteTrack multi-object tracker wrapper.
Assigns persistent track IDs across frames.
Ultralytics provides ByteTrack natively — we use that integration.
Track IDs survive brief occlusions (handled by ByteTrack internally).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logger import get_logger
from services.inference.detector import Detection

logger = get_logger(__name__)


@dataclass
class Track:
    """A tracked person with persistent ID."""
    track_id: int
    bbox_xyxy: List[float]       # Current bounding box
    confidence: float
    frames_since_seen: int       # 0 = seen this frame
    age_frames: int              # Total frames this track has existed
    centroid: Tuple[float, float]

    @property
    def is_active(self) -> bool:
        return self.frames_since_seen == 0


def _compute_centroid(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


class PersonTracker:
    """
    Wraps YOLO's built-in ByteTrack via ultralytics YOLO.track().
    Falls back to simple IoU-based tracking if ultralytics tracking unavailable.
    """

    def __init__(self):
        self._active_tracks: Dict[int, Track] = {}
        self._use_yolo_track = True  # Try ultralytics tracking first

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_number: int,
    ) -> List[Track]:
        """
        Update tracker with new detections.
        Returns list of currently active tracks.
        """
        from services.inference.model_loader import get_models
        models = get_models()

        if models.detector is None or not self._use_yolo_track:
            return self._fallback_update(detections, frame_number)

        try:
            from core.config import get_settings
            settings = get_settings()

            # Use YOLO's built-in ByteTrack
            track_results = models.detector.track(
                frame,
                classes=[0],  # Person only
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                device=settings.inference.device,
            )

            tracks = []
            self._active_tracks.clear()

            if track_results and track_results[0].boxes is not None:
                boxes = track_results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue
                    tid = int(box.id[0])
                    bbox = [float(v) for v in box.xyxy[0]]
                    conf = float(box.conf[0])
                    track = Track(
                        track_id=tid,
                        bbox_xyxy=bbox,
                        confidence=conf,
                        frames_since_seen=0,
                        age_frames=1,
                        centroid=_compute_centroid(bbox),
                    )
                    self._active_tracks[tid] = track
                    tracks.append(track)

            return tracks

        except Exception as exc:
            logger.warning(
                "yolo_tracker_failed",
                error=str(exc),
                fallback="iou_tracking",
            )
            self._use_yolo_track = False
            return self._fallback_update(detections, frame_number)

    def _fallback_update(
        self,
        detections: List[Detection],
        frame_number: int,
    ) -> List[Track]:
        """
        Simple fallback tracker using nearest-centroid matching.
        Not as robust as ByteTrack but works without ultralytics tracking.
        """
        if not detections:
            # Age out all tracks
            for tid in list(self._active_tracks.keys()):
                self._active_tracks[tid].frames_since_seen += 1
                if self._active_tracks[tid].frames_since_seen > 30:
                    del self._active_tracks[tid]
            return list(self._active_tracks.values())

        new_centroids = [_compute_centroid(d.bbox_xyxy) for d in detections]
        matched_track_ids = set()
        unmatched_det_indices = list(range(len(detections)))

        # Match existing tracks to new detections by nearest centroid
        for tid, track in self._active_tracks.items():
            best_idx, best_dist = None, float("inf")
            for i in unmatched_det_indices:
                cx, cy = new_centroids[i]
                tx, ty = track.centroid
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < best_dist and dist < 100:  # pixel threshold
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                det = detections[best_idx]
                track.bbox_xyxy = det.bbox_xyxy
                track.centroid = new_centroids[best_idx]
                track.confidence = det.confidence
                track.frames_since_seen = 0
                track.age_frames += 1
                det.track_id = tid
                matched_track_ids.add(tid)
                unmatched_det_indices.remove(best_idx)

        # Create new tracks for unmatched detections
        next_id = max(self._active_tracks.keys(), default=0) + 1
        for i in unmatched_det_indices:
            det = detections[i]
            track = Track(
                track_id=next_id,
                bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence,
                frames_since_seen=0,
                age_frames=1,
                centroid=new_centroids[i],
            )
            det.track_id = next_id
            self._active_tracks[next_id] = track
            next_id += 1

        # Age out unmatched tracks
        for tid in list(self._active_tracks.keys()):
            if tid not in matched_track_ids and self._active_tracks[tid].frames_since_seen > 0:
                self._active_tracks[tid].frames_since_seen += 1
                if self._active_tracks[tid].frames_since_seen > 30:
                    del self._active_tracks[tid]

        return [t for t in self._active_tracks.values() if t.frames_since_seen == 0]

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._active_tracks.get(track_id)

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self._active_tracks.values() if t.is_active)
