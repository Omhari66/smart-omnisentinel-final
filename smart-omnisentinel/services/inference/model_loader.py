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
    classifier: object = None        # PyTorch temporal CNN+GRU
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
    # 1. Person detector (YOLOv8n)
    # -----------------------------------------------------------------------
    try:
        from ultralytics import YOLO
        detector_path = settings.inference.detector_model
        logger.info("loading_detector", path=detector_path)
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
        logger.info("loading_pose_estimator", path=pose_path)
        loaded.pose_estimator = YOLO(pose_path)
        loaded.pose_estimator.predict(dummy, verbose=False, device=device)
        logger.info("pose_estimator_loaded")
    except Exception as exc:
        logger.warning("pose_estimator_load_failed", error=str(exc))
        # Non-fatal: fall detection can still use geometric rules

    # -----------------------------------------------------------------------
    # 3. Temporal classifier (CNN+GRU)
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
                note="Running without temporal classifier — fall+geometric rules only",
            )
    else:
        logger.warning(
            "classifier_checkpoint_not_found",
            path=checkpoint_path,
            note="Train the model first: python -m ml.training.train_classifier",
        )

    _models = loaded
    logger.info(
        "all_models_loaded",
        has_detector=loaded.detector is not None,
        has_pose=loaded.pose_estimator is not None,
        has_classifier=loaded.classifier is not None,
    )
    return _models
