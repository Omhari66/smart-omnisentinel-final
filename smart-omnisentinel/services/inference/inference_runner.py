"""
services/inference/inference_runner.py
Full pipeline orchestrator per camera.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from core.constants import EventType
from core.event_bus import get_event_bus
from core.logger import get_logger
from services.inference.detector import PersonDetector
from services.inference.fall_detector import FallDetector
from services.inference.frame_buffer import get_buffer
from services.inference.pose_estimator import PoseEstimate, PoseEstimator
from services.inference.temporal_classifier import ClassificationResult, TemporalClassificationEngine
from services.inference.tracker import PersonTracker, Track

logger = get_logger(__name__)
OPTICAL_FLOW_SAMPLE_INTERVAL = 4


@dataclass
class EventCandidate:
    camera_id: str
    event_type: EventType
    raw_confidence: float
    people_count: int
    track_ids: List[int]
    frame_number: int
    timestamp: float
    class_probs: Dict[str, float] = field(default_factory=dict)
    motion_magnitude: float = 0.0


class InferenceRunner:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._detector = PersonDetector()
        self._tracker = PersonTracker()
        self._pose_estimator = PoseEstimator()
        self._fall_detector = FallDetector()
        self._classifier = TemporalClassificationEngine(camera_id=self.camera_id)
        self._frame_number = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._bus = get_event_bus()
        self._frame_buffer = get_buffer(camera_id)

    async def process_frame(self, frame: np.ndarray) -> None:
        loop = asyncio.get_event_loop()
        candidates = await loop.run_in_executor(None, self._process_frame_sync, frame)
        for candidate in candidates:
            await self._bus.publish(
                f"event_candidates.{self.camera_id}",
                self._candidate_to_dict(candidate),
            )

    def _process_frame_sync(self, frame: np.ndarray) -> List[EventCandidate]:
        self._frame_number += 1
        timestamp = time.time()
        candidates: List[EventCandidate] = []
        self._frame_buffer.push(frame)

        detections = self._detector.detect(frame, self._frame_number, timestamp)
        if not detections:
            return []
        tracks = self._tracker.update(detections, frame, self._frame_number)
        if not tracks:
            return []
        poses: Dict[int, PoseEstimate] = self._pose_estimator.estimate(frame, tracks)
        flow_magnitude = self._compute_optical_flow_magnitude(frame)

        for track in tracks:
            pose = poses.get(track.track_id)
            if pose is None or not pose.is_valid:
                continue
            fall_result = self._fall_detector.detect(pose)
            if fall_result.is_fall:
                candidates.append(EventCandidate(
                    camera_id=self.camera_id, event_type=EventType.FALL_COLLAPSE,
                    raw_confidence=fall_result.confidence, people_count=1,
                    track_ids=[track.track_id], frame_number=self._frame_number,
                    timestamp=timestamp, motion_magnitude=flow_magnitude,
                ))

        cls_result = self._classifier.update(tracks=tracks, poses=poses,
                                              frame_optical_flow_magnitude=flow_magnitude)
        if cls_result and cls_result.event_type:
            candidates.append(EventCandidate(
                camera_id=self.camera_id, event_type=cls_result.event_type,
                raw_confidence=cls_result.confidence, people_count=cls_result.people_count,
                track_ids=cls_result.track_ids, frame_number=self._frame_number,
                timestamp=timestamp, class_probs=cls_result.class_probs,
                motion_magnitude=flow_magnitude,
            ))

        crowd = self._check_crowd_aggression(tracks, flow_magnitude, timestamp)
        if crowd:
            candidates.append(crowd)
        return candidates

    def _compute_optical_flow_magnitude(self, frame: np.ndarray) -> float:
        if self._frame_number % OPTICAL_FLOW_SAMPLE_INTERVAL != 0:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
        except Exception:
            magnitude = 0.0
        self._prev_gray = gray
        return magnitude

    def _check_crowd_aggression(self, tracks, flow_magnitude, timestamp):
        if len(tracks) < 4 or flow_magnitude < 3.5:
            return None
        people_factor = min(len(tracks) / 8.0, 1.0)
        motion_factor = min(flow_magnitude / 8.0, 1.0)
        confidence = 0.4 * people_factor + 0.4 * motion_factor + 0.2
        if confidence < 0.50:
            return None
        return EventCandidate(
            camera_id=self.camera_id, event_type=EventType.CROWD_AGGRESSION,
            raw_confidence=confidence, people_count=len(tracks),
            track_ids=[t.track_id for t in tracks], frame_number=self._frame_number,
            timestamp=timestamp, motion_magnitude=flow_magnitude,
        )

    @staticmethod
    def _candidate_to_dict(c: EventCandidate) -> dict:
        return {
            "camera_id": c.camera_id, "event_type": c.event_type.value,
            "raw_confidence": c.raw_confidence, "people_count": c.people_count,
            "track_ids": c.track_ids, "frame_number": c.frame_number,
            "timestamp": c.timestamp, "class_probs": c.class_probs,
            "motion_magnitude": c.motion_magnitude,
        }
