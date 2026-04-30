"""
tests/fixtures/mock_camera_stream.py
---------------------------------------
Mock camera stream for integration tests.
Generates synthetic frames on demand, eliminating the need for
a real RTSP stream during testing.

Usage:
    stream = MockCameraStream(width=640, height=480, fps=4)
    for frame in stream.frames(count=100):
        await runner.process_frame(frame)
"""
from __future__ import annotations

import time
from typing import Generator, Iterator

import numpy as np


class MockCameraStream:
    """
    Generates synthetic video frames for testing.
    Supports several scene modes to trigger different detectors.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 4,
        scene_mode: str = "random",   # "random" | "two_people" | "crowd" | "fall"
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.scene_mode = scene_mode
        self._frame_number = 0

    def next_frame(self) -> np.ndarray:
        """Generate the next synthetic frame."""
        self._frame_number += 1

        if self.scene_mode == "random":
            return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        elif self.scene_mode == "two_people":
            return self._make_two_people_frame()

        elif self.scene_mode == "crowd":
            return self._make_crowd_frame()

        elif self.scene_mode == "fall":
            return self._make_fall_frame()

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def frames(self, count: int) -> Generator[np.ndarray, None, None]:
        """Yield `count` frames."""
        for _ in range(count):
            yield self.next_frame()

    def _make_two_people_frame(self) -> np.ndarray:
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 60
        # Animate two rectangles moving toward each other
        progress = (self._frame_number % 50) / 50.0
        x1 = int(50 + progress * 150)
        x2 = int(500 - progress * 150)
        import cv2
        cv2.rectangle(frame, (x1, 100), (x1 + 70, 380), (180, 100, 60), -1)
        cv2.rectangle(frame, (x2, 100), (x2 + 70, 380), (60, 100, 180), -1)
        noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
        return cv2.add(frame, noise)

    def _make_crowd_frame(self) -> np.ndarray:
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 70
        import cv2
        # 6 people moving in random directions
        np.random.seed(self._frame_number)
        for i in range(6):
            x = int(np.random.uniform(50, 550))
            y = int(np.random.uniform(100, 350))
            cv2.rectangle(frame, (x, y), (x + 50, y + 150), (
                int(np.random.uniform(100, 200)),
                int(np.random.uniform(100, 200)),
                int(np.random.uniform(100, 200))
            ), -1)
        return frame

    def _make_fall_frame(self) -> np.ndarray:
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 60
        import cv2
        # Standing for first 20 frames, then fallen
        if self._frame_number < 20:
            # Standing: tall rectangle
            cv2.rectangle(frame, (280, 100), (360, 400), (180, 100, 60), -1)
        else:
            # Fallen: wide rectangle
            cv2.rectangle(frame, (150, 300), (500, 370), (180, 100, 60), -1)
        return frame
