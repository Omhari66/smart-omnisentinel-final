"""
scripts/benchmark_inference.py
--------------------------------
Measures inference pipeline FPS on current hardware.
Generates random frames and runs the full detection + pose + classification chain.
Use this before deployment to verify hardware meets latency requirements.

Usage:
    python scripts/benchmark_inference.py --frames 200 --device cpu
    python scripts/benchmark_inference.py --frames 200 --device cuda
    python scripts/benchmark_inference.py --resolution 480p

Target benchmarks:
    CPU (modern laptop):     6–10 fps at 480p
    GPU (RTX 3060):         20–30 fps at 720p
    Jetson Orin NX (FP16):  12–18 fps at 480p
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESOLUTIONS = {
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def benchmark(num_frames: int, device: str, resolution: str) -> None:
    w, h = RESOLUTIONS.get(resolution, (854, 480))
    print(f"\nSmartOmniSentinel — Inference Benchmark")
    print(f"  Device:     {device}")
    print(f"  Resolution: {resolution} ({w}x{h})")
    print(f"  Frames:     {num_frames}")
    print()

    # Override device in settings
    os.environ["INFERENCE__DEVICE"] = device
    from core.config import get_settings
    get_settings.cache_clear()

    print("Loading models...")
    from services.inference.model_loader import load_models
    models = load_models()

    from services.inference.detector import PersonDetector
    from services.inference.pose_estimator import PoseEstimator
    from services.inference.tracker import PersonTracker

    detector = PersonDetector()
    tracker = PersonTracker()
    pose_estimator = PoseEstimator()

    # Generate synthetic frames with random noise (simulates real video)
    frames = [
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for _ in range(min(num_frames, 20))  # Pre-generate pool, reuse
    ]

    print(f"Running {num_frames} frame benchmark...\n")

    # Warm up (first few frames are slower due to JIT)
    for frame in frames[:5]:
        detector.detect(frame, 0, time.time())

    # Benchmark
    stage_times = {"detect": [], "track": [], "pose": []}
    total_start = time.perf_counter()

    for i in range(num_frames):
        frame = frames[i % len(frames)]

        t0 = time.perf_counter()
        detections = detector.detect(frame, i, time.time())
        stage_times["detect"].append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        tracks = tracker.update(detections, frame, i)
        stage_times["track"].append(time.perf_counter() - t1)

        t2 = time.perf_counter()
        if tracks:
            pose_estimator.estimate(frame, tracks)
        stage_times["pose"].append(time.perf_counter() - t2)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - total_start
            fps = (i + 1) / elapsed
            print(f"  {i+1:4d}/{num_frames} frames — {fps:.1f} fps")

    total_elapsed = time.perf_counter() - total_start
    overall_fps = num_frames / total_elapsed

    print()
    print("─" * 50)
    print(f"Results ({num_frames} frames @ {resolution} on {device})")
    print("─" * 50)
    print(f"  Overall FPS:        {overall_fps:.1f}")
    print(f"  Total time:         {total_elapsed:.2f}s")
    print()
    print("Per-stage latency (mean ± std, ms):")

    for stage, times in stage_times.items():
        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        p95_ms = np.percentile(times, 95) * 1000
        print(f"  {stage:<10s}: {mean_ms:6.2f} ± {std_ms:5.2f} ms  (p95: {p95_ms:.2f} ms)")

    print()
    print("Assessment:")
    if overall_fps >= 15:
        print("  ✓ EXCELLENT — suitable for 720p real-time deployment")
    elif overall_fps >= 8:
        print("  ✓ GOOD — suitable for 480p real-time deployment")
    elif overall_fps >= 4:
        print("  ⚠  MARGINAL — usable at reduced FPS; consider frame skipping")
    else:
        print("  ✗ INSUFFICIENT — GPU required or reduce resolution/FPS target")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference pipeline FPS")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--resolution", type=str, default="480p",
                        choices=list(RESOLUTIONS.keys()))
    args = parser.parse_args()
    benchmark(args.frames, args.device, args.resolution)
