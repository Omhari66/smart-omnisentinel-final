"""
scripts/replay_video_file.py
------------------------------
Developer tool: feed a recorded video file through the full inference pipeline.
Useful for testing before a real RTSP camera is available.
Does NOT require a running API or database — runs pipeline in isolation.

Usage:
    python scripts/replay_video_file.py --video path/to/test.mp4 --camera-id test-cam-001
    python scripts/replay_video_file.py --video 0  # Use webcam device 0

Output:
    Prints detected events to stdout.
    Saves event log to scripts/output/replay_events.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def replay(video_path: str, camera_id: str, max_frames: int = 0) -> None:
    import cv2

    print(f"SmartOmniSentinel — Video Replay")
    print(f"  Source:    {video_path}")
    print(f"  Camera ID: {camera_id}")
    print(f"  Max frames: {'unlimited' if max_frames == 0 else max_frames}")
    print()

    # Load models
    print("Loading models...")
    try:
        from services.inference.model_loader import load_models
        models = load_models()
        print(f"  Detector:    {'✓' if models.detector else '✗ (not loaded)'}")
        print(f"  Pose:        {'✓' if models.pose_estimator else '✗ (not loaded)'}")
        print(f"  Classifier:  {'✓' if models.classifier else '✗ (using rules only)'}")
    except Exception as exc:
        print(f"  Model load failed: {exc}")
        print("  Continuing with available models...")
    print()

    # Create inference runner
    from services.inference.inference_runner import InferenceRunner
    from core.event_bus import get_event_bus

    runner = InferenceRunner(camera_id=camera_id)
    bus = get_event_bus()

    # Collect events
    events = []
    event_queue = bus.subscribe(f"event_candidates.{camera_id}")

    # Open video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if max_frames == 0 else max_frames
    target_fps = 4  # Effective FPS to process
    skip = max(1, int(source_fps / target_fps))

    print(f"Video info: {source_fps:.1f} fps, processing every {skip}th frame (~{target_fps} fps)")
    print("─" * 60)

    frame_count = 0
    processed = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        # Skip frames to match target FPS
        if frame_count % skip != 0:
            continue

        processed += 1
        await runner.process_frame(frame)

        # Drain event queue
        while not event_queue.empty():
            event = event_queue.get_nowait()
            events.append(event)
            print(
                f"[{processed:04d}] EVENT: {event['event_type']:<25s} "
                f"conf={event['raw_confidence']:.3f} "
                f"people={event['people_count']}"
            )

        if processed % 50 == 0:
            elapsed = time.time() - t_start
            fps = processed / elapsed
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"       Progress: {pct:.1f}% | {fps:.1f} fps processed")

    cap.release()
    elapsed = time.time() - t_start

    print()
    print("─" * 60)
    print(f"Replay complete: {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps)")
    print(f"Total event candidates: {len(events)}")

    if events:
        print("\nEvent summary:")
        from collections import Counter
        counts = Counter(e["event_type"] for e in events)
        for etype, count in counts.most_common():
            avg_conf = sum(
                e["raw_confidence"] for e in events if e["event_type"] == etype
            ) / count
            print(f"  {etype:<30s}: {count:4d} candidates  avg_conf={avg_conf:.3f}")

    # Save event log
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "replay_events.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "source": video_path,
                "camera_id": camera_id,
                "frames_processed": processed,
                "elapsed_seconds": round(elapsed, 2),
                "timestamp": datetime.utcnow().isoformat(),
                "events": events,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nEvent log saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay video through inference pipeline")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file or webcam index (e.g. 0)")
    parser.add_argument("--camera-id", type=str, default="replay-cam-001",
                        help="Camera ID to use in event metadata")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0 = unlimited)")
    args = parser.parse_args()
    asyncio.run(replay(args.video, args.camera_id, args.max_frames))
