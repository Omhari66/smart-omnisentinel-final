"""
api/routes/cameras.py
---------------------
Camera registration, configuration, stream control, and health endpoints.
Stream start/stop delegates to the ingestion service via the event bus.
"""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import CurrentUser, DBSession, Pagination, require_role
from api.schemas.camera import (
    CameraCreate,
    CameraHealthResponse,
    CameraResponse,
    CameraUpdate,
)
from api.schemas.common import MessageResponse, PaginatedResponse
from core.constants import CameraStatus, UserRole
from core.event_bus import get_event_bus
from core.exceptions import NotFoundError
from core.logger import get_logger
from db.models.camera import Camera
from db.models.threshold_profile import ThresholdProfile
from db.models.zone_config import ZoneConfig

router = APIRouter(prefix="/cameras", tags=["cameras"])
logger = get_logger(__name__)


async def _get_camera_or_404(camera_id: uuid.UUID, db: AsyncSession) -> Camera:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return camera


@router.get("", response_model=PaginatedResponse[CameraResponse])
async def list_cameras(
    db: DBSession,
    current_user: CurrentUser,
    pagination: Pagination,
    status_filter: CameraStatus | None = None,
):
    """List all registered cameras with optional status filter."""
    query = select(Camera)
    if status_filter:
        query = query.where(Camera.status == status_filter)

    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar_one()

    cameras_result = await db.execute(
        query.order_by(Camera.created_at.desc())
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    cameras = list(cameras_result.scalars().all())

    return PaginatedResponse[CameraResponse].build(
        items=[CameraResponse.model_validate(c) for c in cameras],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.post("", response_model=CameraResponse, status_code=201)
async def register_camera(
    body: CameraCreate,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Register a new camera. Admin only."""
    # Validate zone and threshold profile if provided
    if body.zone_id:
        zone = await db.get(ZoneConfig, body.zone_id)
        if zone is None:
            raise HTTPException(status_code=404, detail="Zone not found.")

    if body.threshold_profile_id:
        profile = await db.get(ThresholdProfile, body.threshold_profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Threshold profile not found.")

    camera = Camera(
        name=body.name,
        stream_url=body.stream_url,
        location=body.location,
        zone_id=body.zone_id,
        threshold_profile_id=body.threshold_profile_id,
        fps_target=body.fps_target,
        notes=body.notes,
        status=CameraStatus.DISABLED,
    )
    db.add(camera)
    await db.flush()
    await db.refresh(camera)

    logger.info("camera_registered", camera_id=str(camera.id), name=camera.name)
    return CameraResponse.model_validate(camera)


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: uuid.UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    camera = await _get_camera_or_404(camera_id, db)
    return CameraResponse.model_validate(camera)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: uuid.UUID,
    body: CameraUpdate,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    camera = await _get_camera_or_404(camera_id, db)

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(camera, field, value)

    await db.flush()
    await db.refresh(camera)
    logger.info("camera_updated", camera_id=str(camera_id))
    return CameraResponse.model_validate(camera)


@router.delete("/{camera_id}", response_model=MessageResponse)
async def deregister_camera(
    camera_id: uuid.UUID,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    camera = await _get_camera_or_404(camera_id, db)
    if camera.status == CameraStatus.ACTIVE:
        raise HTTPException(
            status_code=409,
            detail="Stop the stream before deregistering.",
        )
    await db.delete(camera)
    logger.info("camera_deregistered", camera_id=str(camera_id))
    return MessageResponse(message="Camera deregistered.")


@router.post("/{camera_id}/start", response_model=MessageResponse)
async def start_stream(
    camera_id: uuid.UUID,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """Instruct the ingestion service to start consuming this camera's stream."""
    camera = await _get_camera_or_404(camera_id, db)

    if camera.status == CameraStatus.ACTIVE:
        return MessageResponse(message="Stream already active.")

    # Publish start command to ingestion service via event bus
    bus = get_event_bus()
    await bus.publish(
        "ingestion.commands",
        {
            "command": "START_STREAM",
            "camera_id": str(camera_id),
            "stream_url": camera.stream_url,
            "fps_target": camera.fps_target,
        },
    )

    # Optimistically update status — ingestion service will confirm or revert
    camera.status = CameraStatus.RECONNECTING
    await db.flush()

    logger.info("stream_start_requested", camera_id=str(camera_id))
    return MessageResponse(message="Stream start command issued.")


@router.post("/{camera_id}/stop", response_model=MessageResponse)
async def stop_stream(
    camera_id: uuid.UUID,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """Instruct the ingestion service to stop consuming this camera's stream."""
    camera = await _get_camera_or_404(camera_id, db)

    if camera.status == CameraStatus.DISABLED:
        return MessageResponse(message="Stream already stopped.")

    bus = get_event_bus()
    await bus.publish(
        "ingestion.commands",
        {"command": "STOP_STREAM", "camera_id": str(camera_id)},
    )

    camera.status = CameraStatus.DISABLED
    await db.flush()

    logger.info("stream_stop_requested", camera_id=str(camera_id))
    return MessageResponse(message="Stream stop command issued.")


@router.get("/{camera_id}/health", response_model=CameraHealthResponse)
async def get_camera_health(
    camera_id: uuid.UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """Return live health metrics for a single camera."""
    camera = await _get_camera_or_404(camera_id, db)

    # In V1, health stats come from the DB last_seen_at field.
    # In V2, query the health_monitor service for richer metrics.
    return CameraHealthResponse(
        camera_id=camera.id,
        camera_name=camera.name,
        status=camera.status,
        last_seen_at=camera.last_seen_at,
        avg_fps=None,  # Populated by health_monitor service in V2
        missed_frames=None,
        stream_url_reachable=(camera.status == CameraStatus.ACTIVE),
    )
