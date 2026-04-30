"""
services/inference/pose_estimator.py
--------------------------------------
YOLOv8n-pose keypoint extractor.
Runs pose estimation on each tracked person's crop.
Returns 17-keypoint COCO skeleton per track.
Skips pose if >40% of keypoints have low confidence (handles occlusion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logger import get_logger
from services.inference.tracker import Track

logger = get_logger(__name__)

# COCO 17 keypoint names (index → name)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17
MIN_KEYPOINT_CONFIDENCE = 0.5
MAX_OCCLUDED_FRACTION = 0.4   # Abstain if >40% keypoints are low confidence


@dataclass
class PoseEstimate:
    """Pose estimation result for one tracked person."""
    track_id: int
    keypoints: np.ndarray         # Shape (17, 2): normalized (x, y) in [0,1]
    keypoint_conf: np.ndarray     # Shape (17,): confidence per keypoint
    is_valid: bool                # False if too many keypoints occluded
    bbox_xyxy: List[float]        # Original bounding box for reference

    @property
    def visible_fraction(self) -> float:
        return float(np.mean(self.keypoint_conf >= MIN_KEYPOINT_CONFIDENCE))

    def get_keypoint(self, name: str) -> Optional[Tuple[float, float]]:
        """Return (x, y) for named keypoint, or None if not visible."""
        idx = KEYPOINT_NAMES.index(name)
        if self.keypoint_conf[idx] < MIN_KEYPOINT_CONFIDENCE:
            return None
        return tuple(self.keypoints[idx])


class PoseEstimator:
    """Extracts poses for a list of tracked persons in one forward pass."""

    def __init__(self):
        from core.config import get_settings
        settings = get_settings()
        self._device = settings.inference.device

    def estimate(
        self,
        frame: np.ndarray,
        tracks: List[Track],
    ) -> Dict[int, PoseEstimate]:
        """
        Run pose estimation on all tracks.
        Returns dict: track_id → PoseEstimate.
        Returns empty dict if pose model not loaded.
        """
        from services.inference.model_loader import get_models
        models = get_models()

        if models.pose_estimator is None or not tracks:
            return {}

        results: Dict[int, PoseEstimate] = {}
        h, w = frame.shape[:2]

        try:
            # Run pose on full frame — YOLO pose returns keypoints for all persons
            pose_results = models.pose_estimator.predict(
                frame,
                verbose=False,
                device=self._device,
            )

            if not pose_results or pose_results[0].keypoints is None:
                return {}

            pose_result = pose_results[0]
            kp_data = pose_result.keypoints.xyn.cpu().numpy()  # (N, 17, 2), normalized
            kp_conf = pose_result.keypoints.conf.cpu().numpy()  # (N, 17)
            boxes = pose_result.boxes.xyxy.cpu().numpy() if pose_result.boxes else None

            # Match pose detections to tracks by bounding box IoU
            for i, (kp, conf) in enumerate(zip(kp_data, kp_conf)):
                if boxes is not None and i < len(boxes):
                    pose_bbox = boxes[i].tolist()
                    # Find best matching track by IoU
                    best_track = self._match_bbox_to_track(pose_bbox, tracks)
                else:
                    best_track = tracks[i] if i < len(tracks) else None

                if best_track is None:
                    continue

                occluded_frac = float(np.mean(conf < MIN_KEYPOINT_CONFIDENCE))
                is_valid = occluded_frac <= MAX_OCCLUDED_FRACTION

                results[best_track.track_id] = PoseEstimate(
                    track_id=best_track.track_id,
                    keypoints=kp,
                    keypoint_conf=conf,
                    is_valid=is_valid,
                    bbox_xyxy=best_track.bbox_xyxy,
                )

        except Exception as exc:
            logger.error("pose_estimation_failed", error=str(exc))

        return results

    def _match_bbox_to_track(
        self,
        pose_bbox: List[float],
        tracks: List[Track],
    ) -> Optional[Track]:
        """Find track with highest IoU to the given bounding box."""
        best_track, best_iou = None, 0.0
        for track in tracks:
            iou = self._compute_iou(pose_bbox, track.bbox_xyxy)
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_track = track
        return best_track

    @staticmethod
    def _compute_iou(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)
