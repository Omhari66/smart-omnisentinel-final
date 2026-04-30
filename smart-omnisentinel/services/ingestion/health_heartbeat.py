"""
services/ingestion/health_heartbeat.py
----------------------------------------
Per-camera liveness monitor.
Publishes periodic health metrics to the "camera_health" event bus channel.
The health_monitor service subscribes and aggregates across all cameras.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from core.constants import CameraStatus, WSEventType
from core.event_bus import get_event_bus
from core.logger import get_logger

logger = get_logger(__name__)

HEARTBEAT_INTERVAL = 10.0   # seconds between health pings
OFFLINE_THRESHOLD = 30.0    # seconds without a frame before marking OFFLINE


@dataclass
class CameraHealthStats:
    camera_id: str
    status: CameraStatus = CameraStatus.ACTIVE
    last_frame_at: float = field(default_factory=time.time)
    frames_total: int = 0
    frames_dropped: int = 0
    avg_fps: float = 0.0
    reconnect_count: int = 0

    @property
    def seconds_since_last_frame(self) -> float:
        return time.time() - self.last_frame_at

    @property
    def is_stale(self) -> bool:
        return self.seconds_since_last_frame > OFFLINE_THRESHOLD


class HealthHeartbeat:
    """
    Tracks per-camera health and publishes heartbeat events.
    One instance per camera; runs as an asyncio background task.
    """

    def __init__(self, camera_id: str):
        self.stats = CameraHealthStats(camera_id=camera_id)
        self._bus = get_event_bus()

    def record_frame(self) -> None:
        """Called by StreamReader for each successfully read frame."""
        self.stats.last_frame_at = time.time()
        self.stats.frames_total += 1

    def record_drop(self) -> None:
        """Called when a frame could not be read."""
        self.stats.frames_dropped += 1

    def record_reconnect(self) -> None:
        self.stats.reconnect_count += 1

    def update_fps(self, fps: float) -> None:
        self.stats.avg_fps = fps

    async def run(self) -> None:
        """Background loop: emit heartbeat every HEARTBEAT_INTERVAL seconds."""
        logger.info("heartbeat_started", camera_id=self.stats.camera_id)
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await self._publish()
            except asyncio.CancelledError:
                logger.info("heartbeat_stopped", camera_id=self.stats.camera_id)
                raise
            except Exception as exc:
                logger.warning("heartbeat_error", camera_id=self.stats.camera_id,
                               error=str(exc))

    async def _publish(self) -> None:
        is_stale = self.stats.is_stale
        status = CameraStatus.OFFLINE if is_stale else CameraStatus.ACTIVE

        payload = {
            "camera_id": self.stats.camera_id,
            "status": status.value,
            "last_frame_at": self.stats.last_frame_at,
            "frames_total": self.stats.frames_total,
            "frames_dropped": self.stats.frames_dropped,
            "avg_fps": round(self.stats.avg_fps, 2),
            "reconnect_count": self.stats.reconnect_count,
            "seconds_since_last_frame": round(self.stats.seconds_since_last_frame, 1),
        }
        await self._bus.publish("camera_health", payload)

        if is_stale:
            logger.warning(
                "camera_stale",
                camera_id=self.stats.camera_id,
                seconds=round(self.stats.seconds_since_last_frame, 1),
            )
