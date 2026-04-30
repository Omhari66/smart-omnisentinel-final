"""
api/routes/config_api.py
-------------------------
Runtime inference configuration endpoint.
Allows the dashboard to read and adjust detection thresholds
without restarting the backend.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.dependencies import CurrentUser, require_role
from core.constants import UserRole
from core.logger import get_logger

router = APIRouter(prefix="/config", tags=["config"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Runtime-mutable inference settings (in-memory, survives until restart)
# ---------------------------------------------------------------------------

class _RuntimeInferenceConfig:
    """Singleton holding mutable inference parameters."""
    def __init__(self):
        from core.config import get_settings
        s = get_settings().inference
        self.threshold: float = 0.60
        self.persistence_frames: int = s.persistence_frames
        self.effective_fps: int = s.effective_fps
        self.window_size: int = 16
        self.ema_alpha: float = s.ema_alpha
        self.cooldown_seconds: float = 5.0


_runtime_config: _RuntimeInferenceConfig | None = None


def get_runtime_config() -> _RuntimeInferenceConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = _RuntimeInferenceConfig()
    return _runtime_config


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class InferenceConfigResponse(BaseModel):
    threshold: float = Field(description="Violence confidence minimum (0.0–1.0)")
    persistence_frames: int = Field(description="Consecutive violent frames to trigger alert")
    effective_fps: int = Field(description="Frames processed per second")
    window_size: int = Field(description="Temporal sliding window size")
    ema_alpha: float = Field(description="EMA smoothing factor")
    cooldown_seconds: float = Field(description="Alert cooldown in seconds")


class InferenceConfigUpdate(BaseModel):
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    persistence_frames: int | None = Field(default=None, ge=1, le=30)
    effective_fps: int | None = Field(default=None, ge=1, le=30)
    window_size: int | None = Field(default=None, ge=4, le=64)
    ema_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    cooldown_seconds: float | None = Field(default=None, ge=0.0, le=60.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/inference", response_model=InferenceConfigResponse)
async def get_inference_config(current_user: CurrentUser):
    """Returns the current runtime inference configuration."""
    cfg = get_runtime_config()
    return InferenceConfigResponse(
        threshold=cfg.threshold,
        persistence_frames=cfg.persistence_frames,
        effective_fps=cfg.effective_fps,
        window_size=cfg.window_size,
        ema_alpha=cfg.ema_alpha,
        cooldown_seconds=cfg.cooldown_seconds,
    )


@router.put("/inference", response_model=InferenceConfigResponse)
async def update_inference_config(
    body: InferenceConfigUpdate,
    current_user: CurrentUser,
):
    """
    Update runtime inference parameters.
    Changes take effect immediately for all new detections.
    """
    cfg = get_runtime_config()

    if body.threshold is not None:
        cfg.threshold = body.threshold
    if body.persistence_frames is not None:
        cfg.persistence_frames = body.persistence_frames
    if body.effective_fps is not None:
        cfg.effective_fps = body.effective_fps
    if body.window_size is not None:
        cfg.window_size = body.window_size
    if body.ema_alpha is not None:
        cfg.ema_alpha = body.ema_alpha
    if body.cooldown_seconds is not None:
        cfg.cooldown_seconds = body.cooldown_seconds

    logger.info(
        "inference_config_updated",
        threshold=cfg.threshold,
        persistence_frames=cfg.persistence_frames,
        updated_by=current_user.email,
    )

    return InferenceConfigResponse(
        threshold=cfg.threshold,
        persistence_frames=cfg.persistence_frames,
        effective_fps=cfg.effective_fps,
        window_size=cfg.window_size,
        ema_alpha=cfg.ema_alpha,
        cooldown_seconds=cfg.cooldown_seconds,
    )
