"""
scripts/test_rtsp_stream.py
-----------------------------
Quick connectivity test for an RTSP camera stream.
Reads 10 frames and reports FPS, resolution, and frame quality.

Usage:
    python scripts/test_rtsp_stream.py rtsp://192.168.1.20:554/stream1
    python scripts/test_rtsp_stream.py 0    # Webcam index
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np


def test_stream(url: str, n_frames: int = 30) -> None:
    print(f"\nTesting stream: {url}")
    print("-" * 50)

    cap = cv2.VideoCapture(int(url) if url.isdigit() else url)
    if not cap.isOpened():
        print(f"ERROR: Cannot open stream: {url}")
        print("Possible causes:")
        print("  - Camera offline or wrong IP/port")
        print("  - Wrong RTSP URL format")
        print("  - Firewall blocking port 554")
        sys.exit(1)

    # Stream properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream properties: {width}x{height} @ {src_fps:.1f} FPS")

    # Read frames and measure actual FPS
    frame_times = []
    frame_sizes = []
    errors = 0

    print(f"Reading {n_frames} frames...")
    for i in range(n_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret or frame is None:
            errors += 1
            continue

        frame_times.append(t1 - t0)
        frame_sizes.append(frame.nbytes)

        # Check for blank frames (potential stream issue)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if i < 3:
            print(f"  Frame {i+1}: brightness={brightness:.1f} blur_score={blur:.1f}")

    cap.release()

    if not frame_times:
        print("\nERROR: No frames successfully read")
        sys.exit(1)

    avg_read_ms = sum(frame_times) / len(frame_times) * 1000
    actual_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    avg_frame_kb = sum(frame_sizes) / len(frame_sizes) / 1024

    print(f"\nResults ({len(frame_times)}/{n_frames} frames successful):")
    print(f"  Actual FPS:        {actual_fps:.1f}")
    print(f"  Avg read time:     {avg_read_ms:.1f}ms")
    print(f"  Avg frame size:    {avg_frame_kb:.0f}KB")
    print(f"  Read errors:       {errors}")

    if errors == 0 and actual_fps > 5:
        print("\n✓ Stream is healthy and ready for SmartOmniSentinel")
    elif errors > n_frames * 0.2:
        print("\n⚠  High error rate — stream may be unstable")
    else:
        print("\n✓ Stream connected (some minor issues, will retry on error)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_rtsp_stream.py <url_or_device_index>")
        print("Examples:")
        print("  python scripts/test_rtsp_stream.py 0")
        print("  python scripts/test_rtsp_stream.py rtsp://192.168.1.20:554/stream1")
        sys.exit(1)

    test_stream(sys.argv[1])
