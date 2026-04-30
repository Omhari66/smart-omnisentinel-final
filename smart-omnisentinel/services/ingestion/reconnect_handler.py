"""
services/ingestion/reconnect_handler.py
-----------------------------------------
Handles reconnection logic for failed camera streams.
Uses exponential backoff to avoid hammering unreachable cameras.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from core.logger import get_logger

logger = get_logger(__name__)

MIN_DELAY = 2.0    # seconds
MAX_DELAY = 60.0   # seconds
BACKOFF_FACTOR = 2.0


@dataclass
class ReconnectState:
    camera_id: str
    attempt: int = 0
    last_attempt_at: float = field(default_factory=time.time)
    next_delay: float = MIN_DELAY

    def next_wait(self) -> float:
        """Exponential backoff with jitter, capped at MAX_DELAY."""
        import random
        delay = min(self.next_delay * (BACKOFF_FACTOR ** self.attempt), MAX_DELAY)
        # Add ±10% jitter to avoid thundering herd
        jitter = delay * 0.1 * (random.random() * 2 - 1)
        return max(MIN_DELAY, delay + jitter)

    def record_attempt(self) -> None:
        self.attempt += 1
        self.last_attempt_at = time.time()
        logger.info(
            "reconnect_attempt",
            camera_id=self.camera_id,
            attempt=self.attempt,
            next_delay=self.next_wait(),
        )

    def reset(self) -> None:
        self.attempt = 0
        self.next_delay = MIN_DELAY


class ReconnectHandler:
    """
    Manages reconnect state for multiple cameras.
    Call wait_before_retry() between connection attempts.
    Call reset() on successful connect.
    """

    def __init__(self):
        self._states: dict[str, ReconnectState] = {}

    def _state(self, camera_id: str) -> ReconnectState:
        if camera_id not in self._states:
            self._states[camera_id] = ReconnectState(camera_id=camera_id)
        return self._states[camera_id]

    async def wait_before_retry(self, camera_id: str) -> None:
        """Async sleep for the calculated backoff duration."""
        state = self._state(camera_id)
        wait = state.next_wait()
        state.record_attempt()
        logger.info(
            "reconnect_waiting",
            camera_id=camera_id,
            seconds=round(wait, 1),
            attempt=state.attempt,
        )
        await asyncio.sleep(wait)

    def reset(self, camera_id: str) -> None:
        """Call on successful connection."""
        if camera_id in self._states:
            self._states[camera_id].reset()

    def attempt_count(self, camera_id: str) -> int:
        return self._states.get(camera_id, ReconnectState(camera_id)).attempt
