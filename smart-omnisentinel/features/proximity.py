"""
features/proximity.py
---------------------
Day 2 Deliverable: 6-Dimensional Inter-Person Proximity & Kinematics Vector.

For each track, extracts interaction features with the 2 nearest other tracks:
  [dist_1, vel_1, overlap_1, dist_2, vel_2, overlap_2]

Concatenation:
  34 keypoints + 6 proximity = 40-dim GRU input vector
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class ProximityFeatureExtractor:
    """
    Computes 6-dimensional pairwise relational features for the 2 nearest tracks.
    """

    def __init__(self):
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}

    def extract_track_proximity(
        self,
        target_track_id: int,
        tracks: List[object],
        poses: Dict[int, object],
    ) -> np.ndarray:
        """
        Returns a 6-dim vector: [dist_1, vel_1, overlap_1, dist_2, vel_2, overlap_2]
        """
        prox_vec = np.zeros(6, dtype=np.float32)

        # Find target track
        target_track = next((t for t in tracks if getattr(t, "track_id", None) == target_track_id), None)
        if target_track is None or len(tracks) < 2:
            return prox_vec

        target_c = target_track.centroid
        target_pose = poses.get(target_track_id)

        # Rank all other tracks by Euclidean distance to target
        other_tracks = [t for t in tracks if getattr(t, "track_id", None) != target_track_id]
        sorted_others = sorted(
            other_tracks,
            key=lambda t: math.hypot(t.centroid[0] - target_c[0], t.centroid[1] - target_c[1])
        )

        # Process top-2 nearest neighbors
        for rank, neighbor in enumerate(sorted_others[:2]):
            offset = rank * 3
            nid = neighbor.track_id
            nc = neighbor.centroid

            # 1. Normalized centroid distance (normalized by diagonal ~1000px)
            dist = math.hypot(nc[0] - target_c[0], nc[1] - target_c[1]) / 1000.0
            prox_vec[offset] = float(np.clip(dist, 0.0, 1.0))

            # 2. Relative velocity magnitude (px/frame)
            prev_tc = self._prev_centroids.get(target_track_id, target_c)
            prev_nc = self._prev_centroids.get(nid, nc)
            v_target = (target_c[0] - prev_tc[0], target_c[1] - prev_tc[1])
            v_neighbor = (nc[0] - prev_nc[0], nc[1] - prev_nc[1])
            rel_vel = math.hypot(v_target[0] - v_neighbor[0], v_target[1] - v_neighbor[1]) / 100.0
            prox_vec[offset + 1] = float(np.clip(rel_vel, 0.0, 1.0))

            # 3. Joint overlap ratio (fraction of joints within 0.1 normalized distance)
            neighbor_pose = poses.get(nid)
            overlap_ratio = 0.0
            if (
                target_pose is not None
                and getattr(target_pose, "is_valid", False)
                and neighbor_pose is not None
                and getattr(neighbor_pose, "is_valid", False)
            ):
                kp_target = target_pose.keypoints  # (17, 2)
                kp_neighbor = neighbor_pose.keypoints  # (17, 2)
                dists = np.linalg.norm(kp_target[:, None, :] - kp_neighbor[None, :, :], axis=-1)
                min_per_joint = np.min(dists, axis=1)
                overlap_ratio = float(np.mean(min_per_joint < 0.10))

            prox_vec[offset + 2] = overlap_ratio

        # Update cache
        self._prev_centroids[target_track_id] = target_c
        return prox_vec
