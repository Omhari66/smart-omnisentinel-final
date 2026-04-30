"""
services/ingestion/frame_sampler.py
-------------------------------------
Adaptive frame sampler: decides which frames to process from the raw stream.

Responsibilities:
  - Skip frames based on detection_skip_frames setting
  - Track actual vs target FPS
  - Signal when inference should run on a given frame
  - Provide frame metadata (number, timestamp, source_fps)
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SampleDecision:
    """Result of sampling check for one raw frame."""
    should_process: bool    # True = send to inference
    frame_number: int       # Raw frame counter
    sample_number: int      # Count of sampled (processed) frames
    timestamp: float        # Unix time this frame was read
    source_fps: float       # Estimated source stream FPS


class FrameSampler:
    """
    Determines whether each raw frame should be sent to inference.
    Uses frame counting to achieve a target effective FPS.
    Tracks both raw frame rate and effective processing rate.
    """

    def __init__(self, target_fps: int = 4, source_fps: float = 25.0):
        self.target_fps = target_fps
        self.source_fps = source_fps
        self._skip_every = max(1, round(source_fps / target_fps))
        self._frame_number = 0
        self._sample_number = 0
        # FPS estimation
        self._fps_window_size = 30
        self._frame_times: list[float] = []

    def update_source_fps(self, fps: float) -> None:
        """Called when actual source FPS is detected from stream."""
        if fps > 0:
            self.source_fps = fps
            self._skip_every = max(1, round(fps / self.target_fps))

    def should_sample(self) -> SampleDecision:
        """
        Call once per raw frame read. Returns sampling decision.
        Does NOT require the frame data itself — just advances the counter.
        """
        self._frame_number += 1
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) > self._fps_window_size:
            self._frame_times.pop(0)

        process = (self._frame_number % self._skip_every == 0)
        if process:
            self._sample_number += 1

        # Estimate actual source FPS from timing
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            estimated_fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
        else:
            estimated_fps = self.source_fps

        return SampleDecision(
            should_process=process,
            frame_number=self._frame_number,
            sample_number=self._sample_number,
            timestamp=now,
            source_fps=estimated_fps,
        )

    @property
    def effective_fps(self) -> float:
        """Actual effective FPS being processed."""
        return self.source_fps / self._skip_every if self._skip_every > 0 else 0.0
