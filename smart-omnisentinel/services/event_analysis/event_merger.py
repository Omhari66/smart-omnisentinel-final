"""
services/event_analysis/event_merger.py
-----------------------------------------
Merges duplicate event candidates into a single active incident window.

Problem being solved:
  A fight that lasts 90 seconds will produce hundreds of event candidates.
  Without merging, these become hundreds of separate DB records and alerts.
  The merger ensures one open incident per (camera, event_type) pair,
  updating its metadata as the event evolves.

Merge criteria (ALL must match for a candidate to be merged):
  - Same camera_id
  - Same event_type
  - Time since incident start < MERGE_WINDOW_SECONDS
  - Track ID overlap with original tracks (>0 shared tracks)

Output:
  - MERGED:  update the existing incident (risk score, duration, people count)
  - NEW:     create a new incident record
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.logger import get_logger

logger = get_logger(__name__)

MERGE_WINDOW_SECONDS = 60.0   # Merge candidates within 60s of incident start
MIN_TRACK_OVERLAP = 0         # 0 = merge even if different tracks (same zone/type)


@dataclass
class ActiveIncidentWindow:
    """Tracks an open incident for merge decisions."""
    incident_id: str
    camera_id: str
    event_type: str
    started_at: float
    last_seen_at: float
    peak_risk_score: int
    track_ids: Set[int]
    people_count_max: int
    candidate_count: int = 1

    @property
    def duration_seconds(self) -> float:
        return self.last_seen_at - self.started_at

    def is_expired(self) -> bool:
        """Window expires if no candidate received in MERGE_WINDOW_SECONDS."""
        return time.time() - self.last_seen_at > MERGE_WINDOW_SECONDS

    def overlaps_tracks(self, track_ids: List[int]) -> bool:
        if MIN_TRACK_OVERLAP == 0:
            return True
        return bool(self.track_ids & set(track_ids))

    def update(
        self,
        risk_score: int,
        track_ids: List[int],
        people_count: int,
    ) -> None:
        self.last_seen_at = time.time()
        self.peak_risk_score = max(self.peak_risk_score, risk_score)
        self.track_ids.update(track_ids)
        self.people_count_max = max(self.people_count_max, people_count)
        self.candidate_count += 1


class EventMerger:
    """
    Maintains a registry of open incident windows per (camera, event_type).
    Decides whether each incoming scored incident is NEW or MERGED.
    """

    def __init__(self):
        # (camera_id, event_type) → ActiveIncidentWindow
        self._windows: Dict[Tuple[str, str], ActiveIncidentWindow] = {}

    def _key(self, camera_id: str, event_type: str) -> Tuple[str, str]:
        return (camera_id, event_type)

    def process(
        self,
        incident_id: str,
        camera_id: str,
        event_type: str,
        risk_score: int,
        track_ids: List[int],
        people_count: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if this incident should be merged with an existing one.

        Returns:
            (is_merge, existing_incident_id)
            is_merge=True  → caller should update existing_incident_id
            is_merge=False → caller should create new incident record
        """
        self._evict_expired()
        key = self._key(camera_id, event_type)
        window = self._windows.get(key)

        if window is not None and window.overlaps_tracks(track_ids):
            # Merge into existing window
            existing_id = window.incident_id
            window.update(risk_score, track_ids, people_count)
            logger.debug(
                "event_merged",
                camera_id=camera_id,
                event_type=event_type,
                existing_incident=existing_id,
                candidate_count=window.candidate_count,
                duration_s=round(window.duration_seconds, 1),
            )
            return True, existing_id

        # New incident: create window
        self._windows[key] = ActiveIncidentWindow(
            incident_id=incident_id,
            camera_id=camera_id,
            event_type=event_type,
            started_at=time.time(),
            last_seen_at=time.time(),
            peak_risk_score=risk_score,
            track_ids=set(track_ids),
            people_count_max=people_count,
        )
        logger.info(
            "new_incident_window_opened",
            camera_id=camera_id,
            event_type=event_type,
            incident_id=incident_id,
        )
        return False, None

    def close_window(self, camera_id: str, event_type: str) -> None:
        """Called when incident is resolved or cooldown set."""
        key = self._key(camera_id, event_type)
        window = self._windows.pop(key, None)
        if window:
            logger.info(
                "incident_window_closed",
                camera_id=camera_id,
                event_type=event_type,
                duration_s=round(window.duration_seconds, 1),
                peak_risk=window.peak_risk_score,
                total_candidates=window.candidate_count,
            )

    def get_window(self, camera_id: str, event_type: str) -> Optional[ActiveIncidentWindow]:
        return self._windows.get(self._key(camera_id, event_type))

    def _evict_expired(self) -> None:
        """Remove windows that have been inactive too long."""
        expired = [k for k, w in self._windows.items() if w.is_expired()]
        for key in expired:
            window = self._windows.pop(key)
            logger.debug(
                "incident_window_expired",
                camera_id=window.camera_id,
                event_type=window.event_type,
                duration_s=round(window.duration_seconds, 1),
            )

    @property
    def open_windows(self) -> int:
        return len(self._windows)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_merger: Optional[EventMerger] = None


def get_event_merger() -> EventMerger:
    global _merger
    if _merger is None:
        _merger = EventMerger()
    return _merger
