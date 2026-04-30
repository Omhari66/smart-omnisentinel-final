"""
services/health_monitor/camera_health.py
------------------------------------------
Subscribes to 'camera_health' bus channel.
Aggregates health stats across cameras, updates DB status,
and pushes WebSocket events when cameras go offline or recover.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.constants import CameraStatus, WSEventType
from core.event_bus import get_event_bus
from core.logger import get_logger

logger = get_logger(__name__)


class CameraHealthMonitor:
    """Monitors all cameras and reacts to status changes."""

    def __init__(self):
        self._last_status: Dict[str, str] = {}   # camera_id → last known status
        self._bus = get_event_bus()

    async def run(self) -> None:
        """Subscribe to camera_health channel and process updates indefinitely."""
        logger.info("camera_health_monitor_started")
        await self._bus.consume("camera_health", self._handle_health_update)

    async def _handle_health_update(self, payload: Dict[str, Any]) -> None:
        camera_id = payload.get("camera_id", "")
        new_status_str = payload.get("status", CameraStatus.ACTIVE.value)

        prev_status = self._last_status.get(camera_id)

        if prev_status != new_status_str:
            self._last_status[camera_id] = new_status_str
            await self._on_status_changed(camera_id, prev_status, new_status_str)

        # Update DB last_seen_at if active
        if new_status_str == CameraStatus.ACTIVE.value:
            await self._update_db_status(camera_id, CameraStatus.ACTIVE)

    async def _on_status_changed(
        self, camera_id: str, prev: Optional[str], new: str
    ) -> None:
        """Handle status transition: update DB and push WebSocket event."""
        logger.info(
            "camera_status_changed",
            camera_id=camera_id,
            previous=prev,
            new=new,
        )
        await self._update_db_status(camera_id, CameraStatus(new))
        await self._push_ws_event(camera_id, prev, new)

    async def _update_db_status(self, camera_id: str, status: CameraStatus) -> None:
        try:
            from db.session import get_session_factory
            from db.models.camera import Camera
            from sqlalchemy import update
            import uuid

            factory = get_session_factory()
            async with factory() as db:
                values: dict = {"status": status.value}
                if status == CameraStatus.ACTIVE:
                    values["last_seen_at"] = datetime.now(tz=timezone.utc)
                await db.execute(
                    update(Camera)
                    .where(Camera.id == uuid.UUID(camera_id))
                    .values(**values)
                )
                await db.commit()
        except Exception as exc:
            logger.warning("camera_status_db_update_failed",
                           camera_id=camera_id, error=str(exc))

    async def _push_ws_event(
        self, camera_id: str, prev: Optional[str], new: str
    ) -> None:
        try:
            from api.schemas.ws_events import CameraStatusChangedPayload, make_event
            from api.websocket_manager import get_ws_manager
            import uuid as _uuid
            from datetime import datetime, timezone

            ws_manager = get_ws_manager()
            payload = CameraStatusChangedPayload(
                camera_id=_uuid.UUID(camera_id),
                camera_name=camera_id,
                previous_status=CameraStatus(prev) if prev else CameraStatus.DISABLED,
                new_status=CameraStatus(new),
                reason="STREAM_HEALTH_MONITOR",
                changed_at=datetime.now(tz=timezone.utc),
            )
            event = make_event(WSEventType.CAMERA_STATUS_CHANGED, payload)
            await ws_manager.broadcast(event.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("camera_status_ws_push_failed", error=str(exc))
