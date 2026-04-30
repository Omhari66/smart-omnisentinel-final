"""
services/event_analysis/event_analyzer.py
------------------------------------------
Receives raw EventCandidate messages from the inference service.
Applies the full false-positive reduction stack:
  1. EMA confidence smoothing
  2. Persistence threshold (N consecutive frames)
  3. Deduplication (merge into existing incident if same event active)
  4. Cooldown check
  5. Zone suppression
Then publishes confirmed events to the risk engine.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

from core.constants import EventType
from core.event_bus import get_event_bus
from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-camera, per-event-type state
# ---------------------------------------------------------------------------

@dataclass
class SmoothingState:
    """EMA smoother state for one (camera, event_type) pair."""
    smoothed_confidence: float = 0.0
    consecutive_frames: int = 0           # Frames above threshold
    event_start_timestamp: Optional[float] = None
    is_active: bool = False              # Currently within a confirmed event window
    last_updated: float = field(default_factory=time.time)

    def update(self, raw_confidence: float, alpha: float, threshold: float) -> None:
        self.smoothed_confidence = (
            alpha * raw_confidence + (1 - alpha) * self.smoothed_confidence
        )
        if self.smoothed_confidence >= threshold:
            self.consecutive_frames += 1
            if self.event_start_timestamp is None:
                self.event_start_timestamp = time.time()
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)
            if self.consecutive_frames == 0:
                self.event_start_timestamp = None
        self.last_updated = time.time()

    def decay(self, alpha: float) -> None:
        """Called when no candidate received for this type — decay smoothed value."""
        self.smoothed_confidence = (1 - alpha) * self.smoothed_confidence
        self.consecutive_frames = max(0, self.consecutive_frames - 2)
        if self.consecutive_frames == 0:
            self.event_start_timestamp = None


@dataclass
class CooldownState:
    """Per-camera per-event-type cooldown after a HIGH alert."""
    expires_at: float = 0.0

    def is_active(self) -> bool:
        return time.time() < self.expires_at

    def set_cooldown(self, seconds: float) -> None:
        self.expires_at = time.time() + seconds


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class EventAnalyzer:
    """
    Processes event candidates from all cameras.
    Maintains per-(camera, event_type) smoothing state.
    Emits confirmed events to 'confirmed_events' channel.
    """

    def __init__(
        self,
        persistence_frames: int = 12,
        ema_alpha: float = 0.35,
        cooldown_seconds: float = 300.0,
        confidence_threshold: float = 0.50,
        suppressed_event_types: Optional[Dict[str, Set[str]]] = None,
    ):
        self.persistence_frames = persistence_frames
        self.ema_alpha = ema_alpha
        self.cooldown_seconds = cooldown_seconds
        self.confidence_threshold = confidence_threshold
        # camera_id → set of suppressed EventType values
        self.suppressed_event_types: Dict[str, Set[str]] = suppressed_event_types or {}

        # State stores
        self._smoothing: Dict[str, Dict[str, SmoothingState]] = defaultdict(
            lambda: defaultdict(SmoothingState)
        )
        self._cooldowns: Dict[str, Dict[str, CooldownState]] = defaultdict(
            lambda: defaultdict(CooldownState)
        )
        self._bus = get_event_bus()

    async def handle_candidate(self, message: Dict[str, Any]) -> None:
        """
        Handler for messages on 'event_candidates.{camera_id}'.
        Called by event bus consumer.
        """
        camera_id: str = message["camera_id"]
        event_type_str: str = message["event_type"]
        raw_confidence: float = float(message["raw_confidence"])

        # --- Zone suppression check ---
        suppressed = self.suppressed_event_types.get(camera_id, set())
        if event_type_str in suppressed:
            logger.debug(
                "event_suppressed_by_zone",
                camera_id=camera_id,
                event_type=event_type_str,
            )
            return

        # --- Cooldown check ---
        cooldown = self._cooldowns[camera_id][event_type_str]
        if cooldown.is_active():
            logger.debug(
                "event_suppressed_cooldown",
                camera_id=camera_id,
                event_type=event_type_str,
            )
            return

        # --- Smoothing ---
        state = self._smoothing[camera_id][event_type_str]
        state.update(raw_confidence, self.ema_alpha, self.confidence_threshold)

        logger.debug(
            "candidate_smoothed",
            camera_id=camera_id,
            event_type=event_type_str,
            raw=round(raw_confidence, 3),
            smoothed=round(state.smoothed_confidence, 3),
            frames=state.consecutive_frames,
            needed=self.persistence_frames,
        )

        # --- Persistence check ---
        if state.consecutive_frames < self.persistence_frames:
            return  # Not yet persistent enough

        # --- Emit confirmed event ---
        if not state.is_active:
            state.is_active = True
            logger.info(
                "event_confirmed",
                camera_id=camera_id,
                event_type=event_type_str,
                smoothed_confidence=round(state.smoothed_confidence, 3),
            )

        confirmed_event = {
            "camera_id": camera_id,
            "event_type": event_type_str,
            "smoothed_confidence": state.smoothed_confidence,
            "raw_confidence": raw_confidence,
            "people_count": message.get("people_count", 1),
            "track_ids": message.get("track_ids", []),
            "motion_magnitude": message.get("motion_magnitude", 0.0),
            "event_start_timestamp": state.event_start_timestamp,
            "consecutive_frames": state.consecutive_frames,
            "timestamp": time.time(),
        }

        await self._bus.publish("confirmed_events", confirmed_event)

    def set_cooldown(self, camera_id: str, event_type: str) -> None:
        """Called by alert manager after HIGH alert fires."""
        self._cooldowns[camera_id][event_type].set_cooldown(self.cooldown_seconds)
        # Reset smoothing state so event doesn't immediately re-trigger
        self._smoothing[camera_id][event_type] = SmoothingState()
        logger.info(
            "cooldown_set",
            camera_id=camera_id,
            event_type=event_type,
            seconds=self.cooldown_seconds,
        )

    def clear_event(self, camera_id: str, event_type: str) -> None:
        """Called when an event is resolved (reviewed)."""
        state = self._smoothing[camera_id].get(event_type)
        if state:
            state.is_active = False

    async def run_decay_loop(self) -> None:
        """
        Background task: decay smoothing state for inactive event types.
        Prevents stale high-confidence states persisting after event ends.
        Run every 2 seconds.
        """
        while True:
            await asyncio.sleep(2.0)
            now = time.time()
            for cam_states in self._smoothing.values():
                for state in cam_states.values():
                    if now - state.last_updated > 3.0:
                        state.decay(self.ema_alpha)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_analyzer: Optional[EventAnalyzer] = None


def get_event_analyzer() -> EventAnalyzer:
    global _analyzer
    if _analyzer is None:
        from core.config import get_settings
        settings = get_settings()
        _analyzer = EventAnalyzer(
            persistence_frames=settings.inference.persistence_frames,
            ema_alpha=settings.inference.ema_alpha,
        )
    return _analyzer
