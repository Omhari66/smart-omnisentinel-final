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
FEATURE_DIM = 51                        # 17 keypoints × 3 (x, y, confidence)
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
    joint_attention: Optional[np.ndarray] = None


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
        
        # 3D CNN Model loader — tries ONNX first, falls back to PyTorch .pth
        import os
        from pathlib import Path
        _models_dir = Path(__file__).parent.parent.parent / "models"
        self.onnx_model_path = str(_models_dir / "violence_model.onnx")
        self.onnx_session = None
        self.r3d_torch_model = None

        # 1. Try ONNX
        if os.path.exists(self.onnx_model_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    self.onnx_model_path, providers=["CPUExecutionProvider"]
                )
                logger.info("Loaded RWF-2000 ONNX 3D CNN model.")
            except ImportError:
                logger.warning("onnxruntime not installed. Trying PyTorch fallback.")
            except Exception as e:
                logger.warning(f"ONNX load failed ({e}). Trying PyTorch fallback.")

        # 2. Fallback: PyTorch .pth state dict
        if self.onnx_session is None:
            pth_path = str(_models_dir / "violence_model.pth")
            if os.path.exists(pth_path):
                try:
                    import torch
                    import torch.nn as nn
                    from torchvision.models.video import r3d_18
                    _r3d = r3d_18(weights=None)
                    _r3d.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(_r3d.fc.in_features, 1)
                    )
                    _r3d.load_state_dict(
                        torch.load(pth_path, map_location="cpu", weights_only=True)
                    )
                    _r3d.eval()
                    self.r3d_torch_model = _r3d
                    logger.info("Loaded RWF-2000 PyTorch R3D-18 model (pth fallback).")
                except Exception as e:
                    logger.error(f"PyTorch model load also failed: {e}")

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

        # --- 3D CNN (RWF-2000) PATH — ONNX or PyTorch backend ---
        _has_cnn = (
            getattr(self, 'onnx_session', None) is not None
            or getattr(self, 'r3d_torch_model', None) is not None
        )
        if _has_cnn:
            # Throttle 3D CNN to run every 4 frames to avoid massive lag
            if self._frame_count % 4 != 0:
                return None

            import cv2
            from services.inference.frame_buffer import get_buffer
            
            buffer = get_buffer(self.camera_id)
            if buffer.frame_count < 16:
                return None
                
            all_frames = buffer.snapshot()
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
            
            if self.onnx_session is not None:
                input_name = self.onnx_session.get_inputs()[0].name
                output_name = self.onnx_session.get_outputs()[0].name
                result = self.onnx_session.run([output_name], {input_name: video_tensor})
                prob = float(result[0][0][0])
            else:
                import torch
                t = torch.tensor(video_tensor)
                with torch.no_grad():
                    prob = float(torch.sigmoid(self.r3d_torch_model(t)).item())
            
            if frame_optical_flow_magnitude > 4.5:
                prob *= 0.3
            elif frame_optical_flow_magnitude > 2.5:
                prob *= 0.7
            
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

        # --- SKELETON ST-GCN / TEMPORAL PATH ---
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
        Build 51-dim feature vector:
          [0:34]  = 17 keypoints × (x, y) normalized to [0, 1]
          [34:51] = 17 keypoint confidence scores [0, 1]
        """
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        if pose is not None and pose.is_valid:
            features[0:34] = pose.keypoints.flatten()
            if hasattr(pose, "keypoint_conf") and pose.keypoint_conf is not None:
                features[34:51] = pose.keypoint_conf
            else:
                features[34:51] = 1.0
        return features

    def _classify(
        self,
        window: TrackWindow,
        tracks: List[Track],
        poses: Dict[int, PoseEstimate],
    ) -> Optional[ClassificationResult]:
        from services.inference.model_loader import get_models
        models = get_models()
        try:
            import torch

            # 1. Extract pairwise inter-person proximity features
            prox_features = np.zeros(8, dtype=np.float32)
            if getattr(models, "proximity_extractor", None) is not None:
                prox_obj = models.proximity_extractor.extract(tracks, poses)
                prox_features = prox_obj.features

            raw_window = window.get_window_tensor()  # (16, 51)
            joint_attention = None

            # 2. ST-GCN Inference
            if getattr(models, "stgcn_classifier", None) is not None:
                x = torch.tensor(raw_window, dtype=torch.float32).unsqueeze(0)  # (1, 16, 51)
                p = torch.tensor(prox_features, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = models.stgcn_classifier(x, p)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                # Grad-CAM explainability hook
                if getattr(models, "stgcn_gradcam", None) is not None:
                    try:
                        joint_attention = models.stgcn_gradcam.generate_heatmap(x, p)
                    except Exception:
                        joint_attention = None

            # Legacy CNN+GRU fallback
            elif getattr(models, "classifier", None) is not None:
                # Legacy model expected 34 dims
                legacy_tensor = raw_window[:, :34]
                x = torch.tensor(legacy_tensor, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = models.classifier(x)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            else:
                return None

            prob_violent = float(probs[CLASS_VIOLENT])

            # 3. Solo Activity Suppression:
            # If only 1 person or no close proximity (feature[7] == 0),
            # aggressively suppress false alarms (pushups, stretching, flailing).
            if prox_features[7] == 0.0 and len(tracks) < 2:
                prob_violent *= 0.35

            class_probs = {
                "normal": float(1.0 - prob_violent),
                "violent_interaction": prob_violent,
            }

            # 4. Temporal Smoothing
            buf = self._smoothing_buffers.setdefault(
                window.track_id, deque(maxlen=SMOOTHING_WINDOW)
            )
            predicted_class = CLASS_VIOLENT if prob_violent >= MIN_CLASSIFICATION_CONFIDENCE else CLASS_NORMAL
            buf.append(predicted_class)
            violent_count = sum(1 for p in buf if p == CLASS_VIOLENT)

            if violent_count < SMOOTHING_THRESHOLD:
                return None

            if prob_violent < MIN_CLASSIFICATION_CONFIDENCE:
                return None

            return ClassificationResult(
                event_type=EventType.VIOLENT_INTERACTION,
                confidence=prob_violent,
                class_probs=class_probs,
                track_ids=[t.track_id for t in tracks],
                people_count=len(tracks),
                joint_attention=joint_attention,
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
