"""
services/event_analysis/persistence_checker.py
------------------------------------------------
Enforces the N-frame persistence requirement.
A smoothed confidence must exceed the minimum threshold for at least
PERSISTENCE_FRAMES consecutive frames before the event is confirmed.

This single rule eliminates ~60–70% of false positives in practice.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple


class PersistenceChecker:
    """
    Tracks consecutive frame counts above threshold per (camera_id, event_type).
    Returns True only when the counter reaches required_frames.
    """

    def __init__(self, required_frames: int = 12):
        self.required_frames = required_frames
        # (camera_id, event_type) → consecutive frames above threshold
        self._counters: Dict[Tuple[str, str], int] = defaultdict(int)

    def check(
        self,
        camera_id: str,
        event_type: str,
        smoothed_confidence: float,
        threshold: float,
    ) -> bool:
        """
        Update counter and return True if persistence requirement is met.
        Resets counter to 0 if confidence drops below threshold.
        """
        key = (camera_id, event_type)
        if smoothed_confidence >= threshold:
            self._counters[key] += 1
        else:
            self._counters[key] = 0

        return self._counters[key] >= self.required_frames

    def reset(self, camera_id: str, event_type: str) -> None:
        """Reset after an event is confirmed to avoid re-triggering immediately."""
        key = (camera_id, event_type)
        self._counters.pop(key, None)

    def get_count(self, camera_id: str, event_type: str) -> int:
        return self._counters.get((camera_id, event_type), 0)
