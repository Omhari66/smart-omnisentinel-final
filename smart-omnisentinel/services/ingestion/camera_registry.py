"""
services/ingestion/camera_registry.py
--------------------------------------
Manages the lifecycle of all active StreamReader tasks.
Listens to the "camera_control" event bus channel for START/STOP commands
published by the API when a camera is enabled or disabled.
"""

from __future__ import annotations

import asyncio
from typing import Dict

from core.event_bus import get_event_bus
from core.logger import get_logger
from services.ingestion.stream_reader import StreamReader

logger = get_logger(__name__)


class CameraRegistry:
    """
    Central registry of running StreamReader tasks.
    One asyncio.Task per active camera.
    """

    def __init__(self, effective_fps: int = 4):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._readers: Dict[str, StreamReader] = {}
        self._effective_fps = effective_fps

    async def start_camera(self, camera_id: str, stream_url: str) -> None:
        if camera_id in self._tasks and not self._tasks[camera_id].done():
            logger.warning("camera_already_active", camera_id=camera_id)
            return

        reader = StreamReader(
            camera_id=camera_id,
            stream_url=stream_url,
            effective_fps=self._effective_fps,
        )
        task = asyncio.create_task(
            reader.run(), name=f"stream_{camera_id}"
        )
        self._readers[camera_id] = reader
        self._tasks[camera_id] = task

        logger.info("camera_task_started", camera_id=camera_id, stream_url=stream_url)

    async def stop_camera(self, camera_id: str) -> None:
        reader = self._readers.pop(camera_id, None)
        task = self._tasks.pop(camera_id, None)

        if reader:
            reader.stop()
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info("camera_task_stopped", camera_id=camera_id)

    async def stop_all(self) -> None:
        camera_ids = list(self._tasks.keys())
        for camera_id in camera_ids:
            await self.stop_camera(camera_id)
        logger.info("all_camera_tasks_stopped")

    @property
    def active_cameras(self) -> list[str]:
        return [
            cid for cid, task in self._tasks.items() if not task.done()
        ]

    async def listen_for_control_commands(self) -> None:
        """
        Subscribe to the "camera_control" channel and handle START/STOP commands.
        Run this as a long-lived asyncio task at application startup.
        """
        bus = get_event_bus()
        logger.info("camera_registry_listening_for_commands")

        async def handle_command(msg: dict) -> None:
            action = msg.get("action")
            camera_id = msg.get("camera_id")
            stream_url = msg.get("stream_url", "")

            if action == "START" and camera_id:
                await self.start_camera(camera_id, stream_url)
            elif action == "STOP" and camera_id:
                await self.stop_camera(camera_id)
            else:
                logger.warning("unknown_camera_control_command", msg=msg)

        await bus.consume("camera_control", handle_command)


# Singleton
_registry: CameraRegistry | None = None


def get_camera_registry() -> CameraRegistry:
    global _registry
    if _registry is None:
        from core.config import get_settings
        settings = get_settings()
        _registry = CameraRegistry(effective_fps=settings.inference.effective_fps)
    return _registry
