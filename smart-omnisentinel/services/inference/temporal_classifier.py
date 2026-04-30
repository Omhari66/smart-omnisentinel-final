"""
services/inference/temporal_classifier.py
------------------------------------------
Sliding window CNN+GRU temporal classifier.
Classifies behavior across a window of T=16 frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from core.constants import EventType
from core.logger import get_logger
from services.inference.pose_estimator import PoseEstimate
from services.inference.tracker import Track

logger = get_logger(__name__)

WINDOW_SIZE = 16
FEATURE_DIM = 34                        # Must match training: 17 joints × 2 coords
CLASS_NORMAL = 0
CLASS_VIOLENT = 1
MIN_CLASSIFICATION_CONFIDENCE = 0.50   # Raised from 0.35 to reduce false alarms
SMOOTHING_WINDOW = 8                   # Number of recent frames inspected
SMOOTHING_THRESHOLD = 5               # Frames that must predict violent to fire alert


@dataclass
class ClassificationResult:
    event_type: Optional[EventType]
    confidence: float
    class_probs: Dict[str, float]
    track_ids: List[int]
    people_count: int


class TrackWindow:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self._frames: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)

    def push_features(self, features: np.ndarray) -> None:
        self._frames.append(features.copy())

    def is_ready(self) -> bool:
        return len(self._frames) == WINDOW_SIZE

    def get_window_tensor(self) -> np.ndarray:
        return np.array(list(self._frames), dtype=np.float32)


class TemporalClassificationEngine:
    def __init__(self, camera_id: str = "default"):
        self.camera_id = camera_id
        self._track_windows: Dict[int, TrackWindow] = {}
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}
        # Per-track deque of recent raw class predictions (for temporal smoothing)
        self._smoothing_buffers: Dict[int, Deque[int]] = {}
        self._frame_count = 0
        
        # ONNX Model Path for RWF-2000 3D CNN
        import os
        from pathlib import Path
        self.onnx_model_path = os.path.join(str(Path(__file__).parent.parent.parent), "models", "violence_model.onnx")
        self.onnx_session = None
        
        if os.path.exists(self.onnx_model_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(self.onnx_model_path)
                logger.info("Successfully loaded RWF-2000 ONNX 3D CNN Model!")
            except ImportError:
                logger.warning("onnxruntime not installed. Falling back to skeleton heuristics.")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")

    def _get_window(self, track_id: int) -> TrackWindow:
        if track_id not in self._track_windows:
            self._track_windows[track_id] = TrackWindow(track_id)
        return self._track_windows[track_id]

    def remove_track(self, track_id: int) -> None:
        self._track_windows.pop(track_id, None)
        self._prev_centroids.pop(track_id, None)
        self._smoothing_buffers.pop(track_id, None)

    def update(
        self,
        tracks: List[Track],
        poses: Dict[int, PoseEstimate],
        frame_optical_flow_magnitude: float = 0.0,
    ) -> Optional[ClassificationResult]:
        if not tracks:
            return None

        self._frame_count += 1

        # --- 3D CNN (RWF-2000) PATH ---
        if getattr(self, 'onnx_session', None) is not None:
            # Throttle 3D CNN to run every 4 frames to avoid massive lag
            if self._frame_count % 4 != 0:
                return None

            import cv2
            from services.inference.frame_buffer import get_buffer
            
            buffer = get_buffer(self.camera_id)
            if buffer.frame_count < 16:
                return None
                
            all_frames = buffer.snapshot()
            # Take the 16 most recent frames to preserve natural motion speed
            recent_frames = all_frames[-16:]
            
            sampled_frames = []
            for buffered_frame in recent_frames:
                frame = buffered_frame.frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))
                frame = frame.astype(np.float32) / 255.0
                mean = np.array([0.43216, 0.394666, 0.37645])
                std = np.array([0.22803, 0.22145, 0.216989])
                frame = (frame - mean) / std
                sampled_frames.append(frame)
                
            video_tensor = np.stack(sampled_frames)
            video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))
            video_tensor = np.expand_dims(video_tensor, axis=0).astype(np.float32)
            
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            
            result = self.onnx_session.run([output_name], {input_name: video_tensor})
            prob = float(result[0][0][0])
            
            # Mitigate false positives caused by camera shake
            if frame_optical_flow_magnitude > 4.5:
                # Camera is likely moving; penalize violence confidence heavily
                prob *= 0.3
            elif frame_optical_flow_magnitude > 2.5:
                prob *= 0.7
            
            # Use temporal smoothing to prevent single-frame flickering
            buf = self._smoothing_buffers.setdefault("onnx_global", deque(maxlen=SMOOTHING_WINDOW))
            buf.append(1 if prob > MIN_CLASSIFICATION_CONFIDENCE else 0)
            
            if sum(buf) >= SMOOTHING_THRESHOLD:
                class_probs = {"normal": 1.0 - prob, "violent_interaction": prob}
                return ClassificationResult(
                    event_type=EventType.VIOLENT_INTERACTION,
                    confidence=prob,
                    class_probs=class_probs,
                    track_ids=[t.track_id for t in tracks],
                    people_count=len(tracks),
                )
            return None

        # --- FALLBACK: SKELETON ST-GCN PATH (Pre-Training) ---
        centroids = {t.track_id: t.centroid for t in tracks}
        min_dist, avg_dist = self._compute_inter_person_distances(centroids)

        for track in tracks:
            pose = poses.get(track.track_id)
            features = self._build_features(
                track=track,
                pose=pose,
                min_interperson_dist=min_dist,
                avg_interperson_dist=avg_dist,
                scene_flow_magnitude=frame_optical_flow_magnitude,
            )
            self._prev_centroids[track.track_id] = track.centroid
            self._get_window(track.track_id).push_features(features)

        active_ids = {t.track_id for t in tracks}
        for tid in list(self._track_windows.keys()):
            if tid not in active_ids:
                self.remove_track(tid)

        primary_track = max(tracks, key=lambda t: t.age_frames)
        primary_window = self._get_window(primary_track.track_id)

        if not primary_window.is_ready():
            return None

        return self._classify(window=primary_window, tracks=tracks, poses=poses)

    def _build_features(
        self,
        track: Track,
        pose: Optional[PoseEstimate],
        min_interperson_dist: float,
        avg_interperson_dist: float,
        scene_flow_magnitude: float,
    ) -> np.ndarray:
        """
        Build 34-dim feature vector matching the training data format:
          features[0:34] = 17 keypoints × (x, y) normalized to [0, 1]
        Motion / inter-person / optical-flow features are intentionally
        excluded here — they were not present in the Kaggle training data.
        They will be re-introduced when training on RWF-2000 with the
        full 55-dim feature set.
        """
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        if pose is not None and pose.is_valid:
            features[0:34] = pose.keypoints.flatten()
        return features

    def _classify(
        self,
        window: TrackWindow,
        tracks: List[Track],
        poses: Dict[int, PoseEstimate],
    ) -> Optional[ClassificationResult]:
        from services.inference.model_loader import get_models
        models = get_models()
        if models.classifier is None:
            return None
        try:
            import torch
            x = torch.tensor(window.get_window_tensor(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = models.classifier(x)           # shape: (1, 2)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # shape: (2,)

            class_probs = {
                "normal": float(probs[CLASS_NORMAL]),
                "violent_interaction": float(probs[CLASS_VIOLENT]),
            }

            # ── Temporal smoothing ────────────────────────────────────────────
            # Record raw prediction in a per-track rolling buffer.
            # Alert fires ONLY when SMOOTHING_THRESHOLD of the last
            # SMOOTHING_WINDOW frames predict violence, suppressing isolated
            # false-positive frames and reducing effective FPR.
            buf = self._smoothing_buffers.setdefault(
                window.track_id, deque(maxlen=SMOOTHING_WINDOW)
            )
            buf.append(int(np.argmax(probs)))
            violent_count = sum(1 for p in buf if p == CLASS_VIOLENT)

            if violent_count < SMOOTHING_THRESHOLD:
                return None

            violent_conf = float(probs[CLASS_VIOLENT])
            if violent_conf < MIN_CLASSIFICATION_CONFIDENCE:
                return None

            return ClassificationResult(
                event_type=EventType.VIOLENT_INTERACTION,
                confidence=violent_conf,
                class_probs=class_probs,
                track_ids=[t.track_id for t in tracks],
                people_count=len(tracks),
            )
        except Exception as exc:
            logger.error("classification_error", error=str(exc))
            return None

    @staticmethod
    def _compute_inter_person_distances(
        centroids: Dict[int, Tuple[float, float]]
    ) -> Tuple[float, float]:
        ids = list(centroids.keys())
        if len(ids) < 2:
            return 0.0, 0.0
        distances = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cx1, cy1 = centroids[ids[i]]
                cx2, cy2 = centroids[ids[j]]
                distances.append(((cx1-cx2)**2 + (cy1-cy2)**2)**0.5)
        return min(distances), sum(distances) / len(distances)
