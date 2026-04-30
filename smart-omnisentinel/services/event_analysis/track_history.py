"""
services/event_analysis/track_history.py
------------------------------------------
Per-track behavioral history store.

Purpose: False positive reduction via track consistency check.
A track that was classified as "normal" for the last 30 seconds
requires stronger evidence before flipping to "violent."
This implements the "track consistency" FP reduction layer from the architecture.

Each track accumulates a sliding window of classification decisions.
The history influences the persistence threshold:
  - Long normal history → require more persistence frames to confirm event
  - Long abnormal history → can confirm with fewer frames (already elevated concern)

Also tracks:
  - Velocity history (for fall detection support)
  - Aspect ratio history (for fall detection support)
  - Zone occupancy (which camera zone the track is in)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)

# How many frames of history to maintain per track
HISTORY_WINDOW = 60          # frames (~15 seconds at 4fps)
NORMAL_HISTORY_THRESHOLD = 30  # frames of "normal" before requiring more evidence


@dataclass
class ClassificationFrame:
    """Single frame's classification result for one track."""
    timestamp: float
    predicted_class: str    # "normal" | "violent_interaction" | "fall" | etc.
    confidence: float
    frame_number: int


@dataclass
class TrackHistory:
    """Complete behavioral history for one persistent track."""
    track_id: int
    camera_id: str
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)

    # Rolling classification history
    classifications: Deque[ClassificationFrame] = field(
        default_factory=lambda: deque(maxlen=HISTORY_WINDOW)
    )
    # Running counts for quick queries
    normal_frame_count: int = 0
    abnormal_frame_count: int = 0

    def record(
        self,
        predicted_class: str,
        confidence: float,
        frame_number: int,
    ) -> None:
        """Record one frame's classification result."""
        frame = ClassificationFrame(
            timestamp=time.time(),
            predicted_class=predicted_class,
            confidence=confidence,
            frame_number=frame_number,
        )
        # If deque is full, oldest element is about to be evicted
        if len(self.classifications) == self.classifications.maxlen:
            evicted = self.classifications[0]
            if evicted.predicted_class == "normal":
                self.normal_frame_count -= 1
            else:
                self.abnormal_frame_count -= 1

        self.classifications.append(frame)
        if predicted_class == "normal":
            self.normal_frame_count += 1
        else:
            self.abnormal_frame_count += 1

        self.last_seen_at = time.time()

    @property
    def has_long_normal_history(self) -> bool:
        """True if track has been consistently normal for a long time."""
        return self.normal_frame_count >= NORMAL_HISTORY_THRESHOLD

    @property
    def consistency_multiplier(self) -> float:
        """
        Returns a multiplier for the persistence threshold.
        Long normal history → higher multiplier (needs more evidence).
        Long abnormal history → lower multiplier (less evidence needed).

        Range: 0.5 (rapid confirmation) to 2.0 (strict confirmation required).
        """
        total = len(self.classifications)
        if total == 0:
            return 1.0

        normal_fraction = self.normal_frame_count / total
        # Linear scale: 0% normal → 0.5x, 100% normal → 2.0x
        return 0.5 + (normal_fraction * 1.5)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def seconds_since_seen(self) -> float:
        return time.time() - self.last_seen_at

    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        return self.seconds_since_seen > max_age_seconds


class TrackHistoryStore:
    """
    Manages behavioral history for all currently-tracked persons.
    Keyed by (camera_id, track_id).
    Automatically evicts stale tracks.
    """

    def __init__(self):
        self._histories: Dict[Tuple[str, int], TrackHistory] = {}

    def _key(self, camera_id: str, track_id: int) -> Tuple[str, int]:
        return (camera_id, track_id)

    def get_or_create(self, camera_id: str, track_id: int) -> TrackHistory:
        key = self._key(camera_id, track_id)
        if key not in self._histories:
            self._histories[key] = TrackHistory(
                track_id=track_id,
                camera_id=camera_id,
            )
        return self._histories[key]

    def record(
        self,
        camera_id: str,
        track_id: int,
        predicted_class: str,
        confidence: float,
        frame_number: int,
    ) -> TrackHistory:
        """Record a classification and return the updated history."""
        history = self.get_or_create(camera_id, track_id)
        history.record(predicted_class, confidence, frame_number)
        return history

    def get_consistency_multiplier(
        self,
        camera_id: str,
        track_id: int,
    ) -> float:
        """
        Get the persistence threshold multiplier for this track.
        Returns 1.0 (neutral) if track has no history yet.
        """
        key = self._key(camera_id, track_id)
        history = self._histories.get(key)
        if history is None:
            return 1.0
        return history.consistency_multiplier

    def remove(self, camera_id: str, track_id: int) -> None:
        self._histories.pop(self._key(camera_id, track_id), None)

    def evict_stale(self, max_age_seconds: float = 30.0) -> int:
        """Remove histories for tracks no longer active. Returns count removed."""
        stale_keys = [
            k for k, h in self._histories.items()
            if h.is_stale(max_age_seconds)
        ]
        for key in stale_keys:
            del self._histories[key]
        if stale_keys:
            logger.debug("track_histories_evicted", count=len(stale_keys))
        return len(stale_keys)

    @property
    def active_track_count(self) -> int:
        return len(self._histories)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: Optional[TrackHistoryStore] = None


def get_track_history_store() -> TrackHistoryStore:
    global _store
    if _store is None:
        _store = TrackHistoryStore()
    return _store
