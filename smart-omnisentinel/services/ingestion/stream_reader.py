"""
services/ingestion/stream_reader.py
------------------------------------
Opens a camera stream (RTSP, HTTP, webcam index, or file path) using OpenCV.
Samples frames at the configured effective FPS and publishes them to the
event bus channel  "frames.{camera_id}".

Design notes:
- Runs as a per-camera asyncio task.
- Frame data is NOT sent over the bus as raw bytes — we send a FramePacket
  dataclass with numpy array + metadata. For Redis transport, serialise to
  JPEG bytes before publishing.
- The circular pre-event buffer is updated here before publishing to inference.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from core.event_bus import get_event_bus
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FramePacket:
    """
    Unit of work passed between ingestion and inference.
    frame: H×W×3 uint8 BGR numpy array (OpenCV format).
    """

    camera_id: str
    frame_index: int
    captured_at: datetime
    frame: np.ndarray = field(repr=False)  # Never log the raw array

    def to_bus_message(self) -> dict:
        """Serialise for in-process event bus (frame is passed by reference)."""
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "captured_at": self.captured_at.isoformat(),
            "frame": self.frame,  # numpy array — safe for in-process only
        }


class StreamReader:
    """
    Manages one camera stream connection.
    Call `run()` as an asyncio task; cancel to stop.
    """

    def __init__(
        self,
        camera_id: str,
        stream_url: str,
        effective_fps: int = 4,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.effective_fps = effective_fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_index = 0
        self._reconnect_count = 0

        # Target interval between sampled frames
        self._sample_interval: float = 1.0 / effective_fps

    def _open_capture(self) -> bool:
        """Attempt to open the video capture. Returns True on success."""
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.stream_url)

        # RTSP-specific tuning: reduce buffering to get near-live frames
        if self.stream_url.startswith("rtsp"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.warning(
                "stream_open_failed",
                camera_id=self.camera_id,
                stream_url=self.stream_url,
            )
            return False

        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._skip_every = max(1, round(src_fps / self.effective_fps))
        logger.info(
            "stream_opened",
            camera_id=self.camera_id,
            source_fps=src_fps,
            effective_fps=self.effective_fps,
            skip_every=self._skip_every,
        )
        return True

    async def run(self) -> None:
        """
        Main loop: read frames, sample at effective_fps, publish to bus.
        Handles reconnection transparently.
        """
        self._running = True
        bus = get_event_bus()

        while self._running:
            if not self._open_capture():
                await self._handle_reconnect(bus, reason="STREAM_OPEN_FAILED")
                continue

            self._reconnect_count = 0
            raw_frame_count = 0
            last_publish_time = 0.0

            logger.info("stream_reading_started", camera_id=self.camera_id)

            # Notify health channel that camera is online
            await bus.publish(
                "camera_health",
                {
                    "camera_id": self.camera_id,
                    "status": "ACTIVE",
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                },
            )

            while self._running:
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    logger.warning(
                        "stream_frame_read_failed",
                        camera_id=self.camera_id,
                        raw_frame_count=raw_frame_count,
                    )
                    await self._handle_reconnect(bus, reason="FRAME_READ_FAILED")
                    break  # Break inner loop → reconnect

                raw_frame_count += 1

                # Frame sampling: skip to maintain effective_fps
                if raw_frame_count % self._skip_every != 0:
                    continue

                now = time.monotonic()
                if (now - last_publish_time) < self._sample_interval * 0.8:
                    # We're ahead of schedule; skip to avoid queue buildup
                    continue

                last_publish_time = now
                self._frame_index += 1

                packet = FramePacket(
                    camera_id=self.camera_id,
                    frame_index=self._frame_index,
                    captured_at=datetime.now(tz=timezone.utc),
                    frame=frame.copy(),  # copy so cap can reuse buffer
                )

                await bus.publish(
                    f"frames.{self.camera_id}",
                    packet.to_bus_message(),
                )

                # Yield to event loop after each frame publish
                await asyncio.sleep(0)

        self._cleanup()

    async def _handle_reconnect(self, bus, reason: str) -> None:
        """Notify health channel of offline status, wait, then retry."""
        self._reconnect_count += 1

        await bus.publish(
            "camera_health",
            {
                "camera_id": self.camera_id,
                "status": "RECONNECTING",
                "reason": reason,
                "attempt": self._reconnect_count,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

        if self._reconnect_count > self.max_reconnect_attempts:
            logger.error(
                "stream_max_reconnects_exceeded",
                camera_id=self.camera_id,
                attempts=self._reconnect_count,
            )
            await bus.publish(
                "camera_health",
                {
                    "camera_id": self.camera_id,
                    "status": "OFFLINE",
                    "reason": "MAX_RECONNECTS_EXCEEDED",
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                },
            )
            self._running = False
            return

        logger.info(
            "stream_reconnecting",
            camera_id=self.camera_id,
            delay=self.reconnect_delay,
            attempt=self._reconnect_count,
        )
        await asyncio.sleep(self.reconnect_delay)

    def stop(self) -> None:
        self._running = False

    def _cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("stream_reader_stopped", camera_id=self.camera_id)
