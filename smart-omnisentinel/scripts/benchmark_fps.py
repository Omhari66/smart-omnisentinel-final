"""
scripts/benchmark_fps.py
------------------------
Latency & Throughput (FPS) Profiling Benchmark for SmartOmniSentinel.

Profiles:
  1. YOLOv8n-Pose Keypoint Extractor
  2. ST-GCN Temporal Graph Classifier
  3. End-to-End Pipeline Latency

Usage:
  python scripts/benchmark_fps.py --runs 100 --device cpu
  python scripts/benchmark_fps.py --runs 200 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.st_gcn import STGCNClassifier


def benchmark_pipeline(device: str = "cpu", num_runs: int = 100):
    print(f"\n=======================================================")
    print(f"  SmartOmniSentinel Latency & FPS Profiling Benchmark")
    print(f"  Device: {device.upper()} | Iterations: {num_runs}")
    print(f"=======================================================\n")

    torch_dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Target PyTorch execution device: {torch_dev}")

    # 1. Benchmark ST-GCN Model
    stgcn = STGCNClassifier(in_channels=3, proximity_dim=8)
    stgcn.eval()
    stgcn.to(torch_dev)

    dummy_skeleton = torch.randn(1, 3, 16, 17, device=torch_dev)
    dummy_prox = torch.randn(1, 8, device=torch_dev)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = stgcn(dummy_skeleton, dummy_prox)

    stgcn_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = stgcn(dummy_skeleton, dummy_prox)
        stgcn_times.append((time.perf_counter() - t0) * 1000.0)

    avg_stgcn_ms = np.mean(stgcn_times)
    fps_stgcn = 1000.0 / avg_stgcn_ms

    print(f"\n[1] ST-GCN Temporal Classifier:")
    print(f"    - Avg Latency: {avg_stgcn_ms:.2f} ms")
    print(f"    - Throughput:  {fps_stgcn:.1f} inferences/sec")

    # 2. Benchmark YOLOv8n-Pose (if available)
    try:
        from ultralytics import YOLO
        pose_model = YOLO("yolov8n-pose.pt")
        pose_model.to(torch_dev)
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        _ = pose_model(dummy_img, verbose=False)

        yolo_times = []
        for _ in range(min(num_runs, 50)):
            t0 = time.perf_counter()
            _ = pose_model(dummy_img, verbose=False)
            yolo_times.append((time.perf_counter() - t0) * 1000.0)

        avg_yolo_ms = np.mean(yolo_times)
        fps_yolo = 1000.0 / avg_yolo_ms
        print(f"\n[2] YOLOv8n-Pose Backbone:")
        print(f"    - Avg Latency: {avg_yolo_ms:.2f} ms")
        print(f"    - Throughput:  {fps_yolo:.1f} FPS")

        # Total Pipeline
        total_latency_ms = avg_yolo_ms + avg_stgcn_ms
        effective_fps = 1000.0 / total_latency_ms
        print(f"\n[3] End-to-End Pipeline (Synchronous):")
        print(f"    - Total Frame Latency: {total_latency_ms:.2f} ms")
        print(f"    - Effective Pipeline FPS: {effective_fps:.1f} FPS")
        print(f"    - Real-Time Capability: {'YES (>= 25 FPS)' if effective_fps >= 25 else 'Edge Throttling / TensorRT Recommended'}")

    except Exception as e:
        print(f"\nYOLO benchmark skipped: {e}")

    print("\n=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    benchmark_pipeline(device=args.device, num_runs=args.runs)
