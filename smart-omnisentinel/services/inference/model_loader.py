"""
services/inference/model_loader.py
-----------------------------------
Central model registry.
Loads all ML models once at service startup; provides access to inference modules.
Supports PyTorch (.pt) and ONNX (.onnx) backends.
On Jetson: swap onnxruntime for TensorRT session (see export/export_tensorrt.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.config import get_settings
from core.exceptions import ConfigurationError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedModels:
    """Container for all loaded model handles."""
    detector: object = None          # YOLO model (ultralytics)
    pose_estimator: object = None    # YOLO pose model (ultralytics)
    classifier: object = None        # PyTorch temporal CNN+GRU (legacy)
    stgcn_classifier: object = None  # Spatio-Temporal Graph Conv Network
    stgcn_gradcam: object = None     # Grad-CAM joint attention extractor
    proximity_extractor: object = None # Inter-person relational kinematics
    fall_classifier: object = None   # Binary fall model (PyTorch) — optional
    use_onnx: bool = False


_models: Optional[LoadedModels] = None


def get_models() -> LoadedModels:
    """Return the singleton LoadedModels instance. Must call load_models() first."""
    if _models is None:
        raise ConfigurationError(
            "Models not loaded. Call load_models() at service startup."
        )
    return _models


def load_models() -> LoadedModels:
    """
    Load all models from disk.
    Called once at inference service startup.
    Returns the LoadedModels singleton.
    """
    global _models
    if _models is not None:
        logger.info("models_already_loaded")
        return _models

    settings = get_settings()
    device = settings.inference.device
    logger.info("loading_models", device=device)

    loaded = LoadedModels()

    # -----------------------------------------------------------------------
    # 0. Inter-Person Proximity Extractor
    # -----------------------------------------------------------------------
    try:
        from services.inference.proximity import InterPersonFeatureExtractor
        loaded.proximity_extractor = InterPersonFeatureExtractor()
        logger.info("proximity_extractor_loaded")
    except Exception as exc:
        logger.warning("proximity_extractor_init_failed", error=str(exc))

    # -----------------------------------------------------------------------
    # 1. Person detector (YOLOv8n)
    # -----------------------------------------------------------------------
    try:
        from ultralytics import YOLO
        detector_path = settings.inference.detector_model
        for cand in ["models/yolov8n.engine", "models/yolov8n.onnx", detector_path]:
            if os.path.exists(cand):
                detector_path = cand
                break
        backend_name = "TensorRT engine" if detector_path.endswith(".engine") else ("ONNX runtime" if detector_path.endswith(".onnx") else "PyTorch")
        logger.info("loading_detector", path=detector_path, backend=backend_name)
        loaded.detector = YOLO(detector_path)
        # Warm-up pass
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        loaded.detector.predict(dummy, verbose=False, device=device)
        logger.info("detector_loaded")
    except Exception as exc:
        logger.error("detector_load_failed", error=str(exc))
        raise ConfigurationError(f"Failed to load detector: {exc}") from exc

    # -----------------------------------------------------------------------
    # 2. Pose estimator (YOLOv8n-pose)
    # -----------------------------------------------------------------------
    try:
        from ultralytics import YOLO
        pose_path = settings.inference.pose_model
        for cand in ["models/yolov8n-pose.engine", "models/yolov8n-pose.onnx", "yolov8n-pose.onnx", pose_path]:
            if os.path.exists(cand):
                pose_path = cand
                break
        backend_name = "TensorRT engine" if pose_path.endswith(".engine") else ("ONNX runtime" if pose_path.endswith(".onnx") else "PyTorch")
        logger.info("loading_pose_estimator", path=pose_path, backend=backend_name)
        loaded.pose_estimator = YOLO(pose_path)
        loaded.pose_estimator.predict(dummy, verbose=False, device=device)
        logger.info("pose_estimator_loaded")
    except Exception as exc:
        logger.warning("pose_estimator_load_failed", error=str(exc))

    # -----------------------------------------------------------------------
    # 3. Spatio-Temporal Graph Conv Network (ST-GCN) & Grad-CAM
    # -----------------------------------------------------------------------
    try:
        import torch
        from ml.models.st_gcn import STGCNClassifier
        from ml.explainability.stgcn_gradcam import STGCNGradCAM

        stgcn_model = STGCNClassifier(in_channels=3, proximity_dim=8)
        stgcn_path = "ml/checkpoints/stgcn_best.pt"
        if os.path.exists(stgcn_path):
            state = torch.load(stgcn_path, map_location=device)
            state_dict = state.get("model_state_dict", state)
            stgcn_model.load_state_dict(state_dict)
            logger.info("stgcn_checkpoint_loaded", path=stgcn_path)
        else:
            logger.info("stgcn_initialized_untrained_or_eval_mode")

        stgcn_model.eval()
        stgcn_model.to(device)
        loaded.stgcn_classifier = stgcn_model
        loaded.stgcn_gradcam = STGCNGradCAM(stgcn_model)
        logger.info("stgcn_classifier_and_gradcam_loaded")
    except Exception as exc:
        logger.warning("stgcn_init_failed", error=str(exc))

    # -----------------------------------------------------------------------
    # 4. Temporal classifier fallback (CNN+GRU)
    # -----------------------------------------------------------------------
    checkpoint_path = settings.inference.classifier_checkpoint
    if os.path.exists(checkpoint_path):
        try:
            import torch
            from ml.models.temporal_model import TemporalClassifier

            logger.info("loading_classifier", path=checkpoint_path)
            loaded.classifier = TemporalClassifier()
            state = torch.load(checkpoint_path, map_location=device)
            loaded.classifier.load_state_dict(state["model_state_dict"])
            loaded.classifier.eval()
            loaded.classifier.to(device)
            logger.info("classifier_loaded")
        except Exception as exc:
            logger.warning(
                "classifier_load_failed",
                error=str(exc),
                note="Running without legacy temporal classifier",
            )
    else:
        logger.warning(
            "classifier_checkpoint_not_found",
            path=checkpoint_path,
        )

    _models = loaded
    logger.info(
        "all_models_loaded",
        has_detector=loaded.detector is not None,
        has_pose=loaded.pose_estimator is not None,
        has_stgcn=loaded.stgcn_classifier is not None,
        has_classifier=loaded.classifier is not None,
    )
    return _models
