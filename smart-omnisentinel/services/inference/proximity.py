"""
services/inference/proximity.py
-------------------------------
Pairwise Inter-Person Proximity and Relational Kinematics Engine.

Calculates multi-person relational features to distinguish true interpersonal
violence (grappling, punches, tackles) from energetic solo movements
(e.g., jumping jacks, pushups, rapid gesturing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.inference.pose_estimator import PoseEstimate
from services.inference.tracker import Track

PROXIMITY_FEATURE_DIM = 8


@dataclass
class ProximityFeatures:
    """
    Extracted pairwise features between the two primary interacting tracks.
    """
    features: np.ndarray  # Shape: (8,)
    track_ids: Tuple[Optional[int], Optional[int]]
    min_joint_distance: float
    centroid_distance: float
    bbox_iou: float


class InterPersonFeatureExtractor:
    """
    Extracts geometric and kinematic interaction features between multiple individuals.
    """

    def __init__(self):
        # Cache previous centroids for velocity computation: {track_id: (cx, cy, timestamp/frame)}
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}

    def extract(
        self,
        tracks: List[Track],
        poses: Dict[int, PoseEstimate],
    ) -> ProximityFeatures:
        """
        Extracts an 8-dimensional relational feature vector:
          [0]: Normalized centroid Euclidean distance
          [1]: Rate of closure / approaching velocity (delta distance)
          [2]: Bounding box Intersection-over-Union (IoU)
          [3]: Minimum distance between extremities (wrists/elbows to head/torso)
          [4]: Relative lateral velocity (dx1/dt - dx2/dt)
          [5]: Relative vertical velocity (dy1/dt - dy2/dt)
          [6]: Average torso-to-torso distance
          [7]: Interaction indicator (1.0 if >=2 people in close range, else 0.0)
        """
        feat = np.zeros(PROXIMITY_FEATURE_DIM, dtype=np.float32)

        if len(tracks) < 2:
            # Single person: clean solo-motion prior
            return ProximityFeatures(
                features=feat,
                track_ids=(tracks[0].track_id if tracks else None, None),
                min_joint_distance=1.0,
                centroid_distance=1.0,
                bbox_iou=0.0,
            )

        # Find the two closest tracks
        t1, t2, min_dist = self._find_closest_pair(tracks)
        feat[0] = min_dist

        # Compute rate of closure (approaching speed)
        prev_dist = self._get_previous_distance(t1.track_id, t2.track_id)
        if prev_dist is not None:
            # Negative means getting closer, positive means separating
            feat[1] = prev_dist - min_dist
        else:
            feat[1] = 0.0

        # Update cache
        self._prev_centroids[t1.track_id] = t1.centroid
        self._prev_centroids[t2.track_id] = t2.centroid

        # Bounding box IoU
        bbox1 = getattr(t1, "bbox_xyxy", getattr(t1, "bbox", [0, 0, 0, 0]))
        bbox2 = getattr(t2, "bbox_xyxy", getattr(t2, "bbox", [0, 0, 0, 0]))
        iou = self._compute_iou(bbox1, bbox2)
        feat[2] = iou

        # Joint-level extremity interaction (if poses are present)
        pose1 = poses.get(t1.track_id)
        pose2 = poses.get(t2.track_id)

        min_ext_dist = 1.0
        torso_dist = min_dist

        if pose1 and pose1.is_valid and pose2 and pose2.is_valid:
            kp1 = pose1.keypoints  # (17, 2)
            kp2 = pose2.keypoints  # (17, 2)

            # Wrists (9, 10) & Elbows (7, 8)
            arms1 = kp1[[7, 8, 9, 10]]
            arms2 = kp2[[7, 8, 9, 10]]

            # Head (0..4) & Torso (5, 6, 11, 12)
            core1 = kp1[[0, 1, 2, 5, 6, 11, 12]]
            core2 = kp2[[0, 1, 2, 5, 6, 11, 12]]

            # Distance from person 1's arms to person 2's core & vice-versa
            d1 = np.linalg.norm(arms1[:, None, :] - core2[None, :, :], axis=-1)
            d2 = np.linalg.norm(arms2[:, None, :] - core1[None, :, :], axis=-1)
            min_ext_dist = float(min(np.min(d1), np.min(d2)))

            # Torso distance: Midpoint of shoulders
            torso1 = np.mean(kp1[[5, 6]], axis=0)
            torso2 = np.mean(kp2[[5, 6]], axis=0)
            torso_dist = float(np.linalg.norm(torso1 - torso2))

        feat[3] = min_ext_dist

        # Relative velocity components
        prev_c1 = self._prev_centroids.get(t1.track_id, t1.centroid)
        prev_c2 = self._prev_centroids.get(t2.track_id, t2.centroid)
        v1 = (t1.centroid[0] - prev_c1[0], t1.centroid[1] - prev_c1[1])
        v2 = (t2.centroid[0] - prev_c2[0], t2.centroid[1] - prev_c2[1])
        feat[4] = float(v1[0] - v2[0])
        feat[5] = float(v1[1] - v2[1])
        feat[6] = torso_dist

        # Interaction flag (Close interaction trigger if distance < 0.35 of frame)
        feat[7] = 1.0 if (min_dist < 0.35 or iou > 0.05) else 0.0

        return ProximityFeatures(
            features=feat,
            track_ids=(t1.track_id, t2.track_id),
            min_joint_distance=min_ext_dist,
            centroid_distance=min_dist,
            bbox_iou=iou,
        )

    def _find_closest_pair(self, tracks: List[Track]) -> Tuple[Track, Track, float]:
        min_dist = float("inf")
        best_pair = (tracks[0], tracks[1])
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                c1 = tracks[i].centroid
                c2 = tracks[j].centroid
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (tracks[i], tracks[j])
        return best_pair[0], best_pair[1], min_dist

    def _get_previous_distance(self, tid1: int, tid2: int) -> Optional[float]:
        if tid1 in self._prev_centroids and tid2 in self._prev_centroids:
            c1 = self._prev_centroids[tid1]
            c2 = self._prev_centroids[tid2]
            return math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        return None

    @staticmethod
    def _compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = float(boxAArea + boxBArea - interArea)
        if denominator <= 0:
            return 0.0
        return interArea / denominator
