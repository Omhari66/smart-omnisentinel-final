"""tests/unit/test_fall_detector.py — Geometric fall detection rules."""

import numpy as np
import pytest

from services.inference.fall_detector import FallDetector
from services.inference.pose_estimator import PoseEstimate


def make_pose(
    track_id: int = 1,
    head_y: float = 0.2,    # Normalized (0=top, 1=bottom)
    hip_y: float = 0.55,
    bbox: list = None,
    is_valid: bool = True,
) -> PoseEstimate:
    """Build a PoseEstimate with controllable head and hip positions."""
    if bbox is None:
        bbox = [100, 50, 200, 350]  # Tall bounding box (standing person)

    keypoints = np.zeros((17, 2), dtype=np.float32)
    conf = np.ones(17, dtype=np.float32) * 0.9

    # nose = index 0, left_hip = 11, right_hip = 12
    keypoints[0] = [0.5, head_y]   # nose
    keypoints[11] = [0.45, hip_y]  # left_hip
    keypoints[12] = [0.55, hip_y]  # right_hip

    return PoseEstimate(
        track_id=track_id,
        keypoints=keypoints,
        keypoint_conf=conf,
        is_valid=is_valid,
        bbox_xyxy=bbox,
    )


class TestFallDetectorGeometric:

    def test_standing_person_no_fall(self):
        detector = FallDetector()
        pose = make_pose(bbox=[100, 50, 200, 350])  # height 300 >> width 100

        # Feed many frames of standing person
        for _ in range(15):
            result = detector.detect(pose)

        assert result.is_fall is False
        assert result.geometric_triggered is False

    def test_fallen_person_triggers_after_transition(self):
        """
        Patch get_models() at its source module so the classifier path returns
        0.0 (fall_classifier=None), letting geometric rules alone decide.
        No real ML models needed for this test.
        """
        from unittest.mock import MagicMock, patch

        mock_models = MagicMock()
        mock_models.fall_classifier = None   # Classifier not loaded → geometric only

        with patch("services.inference.model_loader.get_models", return_value=mock_models):
            detector = FallDetector()

            # First: simulate standing frames so was_standing=True
            standing_pose = make_pose(
                bbox=[100, 100, 200, 400],  # height/width = 3.0 → standing
                head_y=0.1,
            )
            for _ in range(12):
                detector.detect(standing_pose)

            # Then: simulate fallen (horizontal bounding box)
            fallen_pose = make_pose(
                bbox=[50, 200, 350, 280],   # width=300, height=80 → aspect≈0.27
                head_y=0.5,
            )
            result = None
            for _ in range(5):
                result = detector.detect(fallen_pose)

        assert result is not None
        assert result.geometric_triggered is True

    def test_invalid_pose_returns_no_fall(self):
        detector = FallDetector()
        pose = make_pose(is_valid=False)
        result = detector.detect(pose)
        assert result.is_fall is False

    def test_different_tracks_have_independent_state(self):
        detector = FallDetector()
        # Track 1: standing history
        standing = make_pose(track_id=1, bbox=[100, 50, 200, 350])
        for _ in range(12):
            detector.detect(standing)

        # Track 2: new track with no standing history → fall shouldn't trigger
        fallen = make_pose(track_id=2, bbox=[50, 200, 350, 280])
        result = detector.detect(fallen)
        # No standing history for track 2 → was_standing=False → no trigger
        assert result.geometric_triggered is False

    def test_remove_track_cleans_state(self):
        detector = FallDetector()
        pose = make_pose(track_id=42)
        detector.detect(pose)
        assert 42 in detector._states
        detector.remove_track(42)
        assert 42 not in detector._states
