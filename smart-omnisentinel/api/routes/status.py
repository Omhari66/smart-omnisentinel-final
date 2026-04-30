"""
api/routes/status.py
---------------------
System status endpoint for the dashboard status panel.
Returns live health metrics without requiring database queries.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from api.dependencies import CurrentUser, DBSession
from core.config import get_settings
from core.logger import get_logger
from db.models.camera import Camera
from db.models.incident import Incident
from core.constants import CameraStatus

router = APIRouter(prefix="/status", tags=["status"])
logger = get_logger(__name__)

# Track server start time for uptime calculation
_server_start_time = time.time()


class SystemStatusResponse(BaseModel):
    uptime_seconds: float = Field(description="Server uptime in seconds")
    active_cameras: int = Field(description="Number of cameras with ACTIVE status")
    total_incidents: int = Field(description="Total incidents in database")
    latest_incident_at: str | None = Field(description="ISO timestamp of most recent incident")
    model_checkpoint: str = Field(description="Active classifier checkpoint file")
    websocket_connections: int = Field(description="Current WebSocket connections")
    inference_device: str = Field(description="cpu or cuda")
    effective_fps: int = Field(description="Configured processing FPS")


@router.get("", response_model=SystemStatusResponse)
async def get_system_status(
    db: DBSession,
    current_user: CurrentUser,
):
    """Returns live system health status for the dashboard."""
    settings = get_settings()

    # Active cameras
    cam_result = await db.execute(
        select(func.count()).select_from(Camera).where(
            Camera.status == CameraStatus.ACTIVE
        )
    )
    active_cameras = cam_result.scalar_one()

    # Total incidents
    total_result = await db.execute(
        select(func.count()).select_from(Incident)
    )
    total_incidents = total_result.scalar_one()

    # Latest incident timestamp
    latest_result = await db.execute(
        select(Incident.detected_at).order_by(Incident.detected_at.desc()).limit(1)
    )
    latest_row = latest_result.scalar_one_or_none()
    latest_str = latest_row.isoformat() if latest_row else None

    # WebSocket connections
    try:
        from api.websocket_manager import get_ws_manager
        ws_count = get_ws_manager().active_count
    except Exception:
        ws_count = 0

    return SystemStatusResponse(
        uptime_seconds=round(time.time() - _server_start_time, 1),
        active_cameras=active_cameras,
        total_incidents=total_incidents,
        latest_incident_at=latest_str,
        model_checkpoint=settings.inference.classifier_checkpoint,
        websocket_connections=ws_count,
        inference_device=settings.inference.device,
        effective_fps=settings.inference.effective_fps,
    )
