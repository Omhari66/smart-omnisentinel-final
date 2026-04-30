"""
services/inference/frame_buffer.py
------------------------------------
Per-camera circular frame buffer.
Retains the last N seconds of raw frames in memory.
When an event is confirmed, the buffer is flushed to disk as the pre-event clip.

Memory estimate: 480p frame (JPEG compressed in buffer) ~15–25KB.
At 4fps × 45s = 180 frames × 20KB ≈ 3.6MB per camera.
Manageable for 4–8 cameras on a single machine.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from core.config import get_settings
from core.constants import PRE_EVENT_BUFFER_SECONDS
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BufferedFrame:
    """A single frame stored in the circular buffer."""
    timestamp: float          # Unix timestamp (time.time())
    frame: np.ndarray         # Raw BGR frame
    frame_number: int


class CameraFrameBuffer:
    """
    Circular buffer for one camera.
    Automatically evicts frames older than PRE_EVENT_BUFFER_SECONDS.
    """

    def __init__(self, camera_id: str, buffer_seconds: int = PRE_EVENT_BUFFER_SECONDS):
        self.camera_id = camera_id
        self.buffer_seconds = buffer_seconds
        self._frames: Deque[BufferedFrame] = collections.deque()
        self._frame_count: int = 0

    def push(self, frame: np.ndarray) -> None:
        """Add a frame; evict frames outside the retention window."""
        self._frame_count += 1
        buffered = BufferedFrame(
            timestamp=time.time(),
            frame=frame.copy(),  # Copy — inference may mutate original
            frame_number=self._frame_count,
        )
        self._frames.append(buffered)
        self._evict_old_frames()

    def _evict_old_frames(self) -> None:
        cutoff = time.time() - self.buffer_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def flush(self) -> List[BufferedFrame]:
        """
        Return all buffered frames and clear the buffer.
        Called when an event is confirmed and the pre-event clip needs to be written.
        """
        frames = list(self._frames)
        self._frames.clear()
        logger.debug(
            "frame_buffer_flushed",
            camera_id=self.camera_id,
            frame_count=len(frames),
        )
        return frames

    def snapshot(self) -> List[BufferedFrame]:
        """Return buffered frames without clearing (for ongoing event monitoring)."""
        return list(self._frames)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def oldest_timestamp(self) -> Optional[float]:
        return self._frames[0].timestamp if self._frames else None


# ---------------------------------------------------------------------------
# Global buffer registry (one buffer per active camera)
# ---------------------------------------------------------------------------

_buffers: Dict[str, CameraFrameBuffer] = {}


def get_buffer(camera_id: str) -> CameraFrameBuffer:
    """Get or create a frame buffer for the given camera."""
    if camera_id not in _buffers:
        _buffers[camera_id] = CameraFrameBuffer(camera_id=camera_id)
        logger.info("frame_buffer_created", camera_id=camera_id)
    return _buffers[camera_id]


def remove_buffer(camera_id: str) -> None:
    """Remove buffer when camera is stopped."""
    if camera_id in _buffers:
        del _buffers[camera_id]
        logger.info("frame_buffer_removed", camera_id=camera_id)
