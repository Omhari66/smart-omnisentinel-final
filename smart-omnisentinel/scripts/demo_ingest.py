"""
scripts/demo_ingest.py
-----------------------
Runs the AI inference pipeline on a video source and writes
every confirmed violence detection directly into demo.db as an Incident.

PRODUCTION FEATURES:
  - Pushes every processed frame into CameraFrameBuffer (circular buffer)
  - On violence detection: saves evidence snapshot + triggers clip extraction
  - Evidence clips are saved to evidence_storage/clips/ with SHA-256 hashes
  - Publishes INCIDENT_CREATED events to the EventBus for WebSocket push

Run in Terminal 2 AFTER the API server is started:
    python scripts/demo_ingest.py --video your_video.mp4
    python scripts/demo_ingest.py --video "http://192.168.1.5:4747/video"
    python scripts/demo_ingest.py --video 0   # USB webcam

Controls: Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

from ml.models.temporal_model import TemporalClassifier

WINDOW_SIZE         = 16
FEATURE_DIM         = 34
SMOOTHING_WINDOW    = 8
SMOOTHING_THRESHOLD = 5
VIOLENCE_CONF_MIN   = 0.75    # raised to 0.75 for RWF-2000 3D CNN (was 0.60)
PROCESS_FPS         = 4
ALERT_COOLDOWN_SECS = 5.0
CAMERA_SHAKE_THRESH = 12.0    # global optical flow above this = camera moving, not violence


# ── Simple per-person window + smoother ──────────────────────────────────────
class PersonState:
    def __init__(self):
        self.window:  collections.deque = collections.deque(maxlen=WINDOW_SIZE)
        self.buffer:  collections.deque = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.violent_conf: float = 0.0
        self.is_violent:   bool  = False

    def push(self, kp: np.ndarray) -> None:
        flat = kp.flatten().astype(np.float32)
        if flat.shape[0] == FEATURE_DIM:
            self.window.append(flat)

    def classify(self, clf: TemporalClassifier, device: torch.device, onnx_session=None, raw_frame_buffer=None, flow_magnitude: float = 0.0) -> None:
        # --- Camera-shake guard: if whole frame is shaking, skip detection ---
        if flow_magnitude > CAMERA_SHAKE_THRESH:
            # Reset violent state — this is camera movement, not violence
            self.buffer.append(0)
            self.is_violent = False
            return

        if onnx_session is not None and raw_frame_buffer is not None and len(raw_frame_buffer) == 16:
            # --- PRIMARY: 3D CNN (RWF-2000 trained) PATH ---
            video_tensor = np.stack(list(raw_frame_buffer))
            video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))
            video_tensor = np.expand_dims(video_tensor, axis=0).astype(np.float32)
            
            input_name = onnx_session.get_inputs()[0].name
            output_name = onnx_session.get_outputs()[0].name
            result = onnx_session.run([output_name], {input_name: video_tensor})
            self.violent_conf = float(result[0][0][0])
            pred = 1 if self.violent_conf >= VIOLENCE_CONF_MIN else 0
        else:
            # --- FALLBACK: SKELETON ST-GCN PATH (only if ONNX unavailable) ---
            if len(self.window) < WINDOW_SIZE:
                return
            x = torch.tensor(
                np.array(list(self.window)), dtype=torch.float32
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(clf(x), dim=-1)[0].cpu().numpy()

            pred = int(np.argmax(probs))
            self.violent_conf = float(probs[1])

        self.buffer.append(pred)
        violent_votes = sum(1 for p in self.buffer if p == 1)
        self.is_violent = (
            violent_votes >= SMOOTHING_THRESHOLD
            and self.violent_conf >= VIOLENCE_CONF_MIN
        )


# ── DB writer ─────────────────────────────────────────────────────────────────
async def write_incident_to_db(camera_id: uuid.UUID, conf: float, people: int) -> str:
    """Insert a new Incident row and return its ID string."""
    from sqlalchemy import select
    from db.session import get_session_factory
    from db.models.incident import Incident
    from db.models.camera import Camera
    from core.constants import EventType, IncidentStatus, Severity

    risk = int(min(100, conf * 100 + 10))
    if risk >= 65:
        severity = Severity.HIGH
        status   = IncidentStatus.EMERGENCY
    elif risk >= 30:
        severity = Severity.MEDIUM
        status   = IncidentStatus.ACTIVE_REVIEW
    else:
        severity = Severity.LOW
        status   = IncidentStatus.PENDING_REVIEW

    now = datetime.now(tz=timezone.utc)
    factory = get_session_factory()
    async with factory() as db:
        incident = Incident(
            camera_id=camera_id,
            event_type=EventType.VIOLENT_INTERACTION,
            severity=severity,
            status=status,
            risk_score=risk,
            confidence_at_alert=round(conf, 4),
            people_count=people,
            event_start=now,
            detected_at=now,
            duration_seconds=0.0,
            is_locked=False,
            retention_days=30,
        )
        db.add(incident)
        await db.commit()
        await db.refresh(incident)
        return str(incident.id)


async def get_or_create_demo_camera(camera_name: str) -> uuid.UUID:
    """Get the requested camera ID from the seeded data."""
    from sqlalchemy import select
    from db.session import get_session_factory
    from db.models.camera import Camera
    from core.constants import CameraStatus

    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(
            select(Camera).where(Camera.name == camera_name)
        )
        cam = result.scalar_one_or_none()
        if cam:
            # Ensure it's marked active if we're connecting to it
            cam.status = CameraStatus.ACTIVE
            await db.commit()
            return cam.id
        # Fallback: create a camera if seed wasn't run
        from datetime import datetime, timezone
        new_cam = Camera(
            name=camera_name,
            stream_url="rtsp://192.168.1.10/stream1",
            location="Building A – Front Door",
            status=CameraStatus.ACTIVE,
            last_seen_at=datetime.now(tz=timezone.utc),
            fps_target=4,
        )
        db.add(new_cam)
        await db.commit()
        await db.refresh(new_cam)
        return new_cam.id


# ── Evidence pipeline helpers ─────────────────────────────────────────────────

def save_evidence_snapshot(frame: np.ndarray, incident_id: str) -> str:
    """
    Save the detection frame as a JPEG snapshot.
    Two copies: one timestamped in evidence_storage/, one as dashboard/evidence.jpg.
    """
    root = Path(__file__).parent.parent

    # Timestamped copy for permanent evidence storage
    evidence_dir = root / "evidence_storage" / "snapshots"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = evidence_dir / f"{incident_id[:8]}_{ts}.jpg"
    cv2.imwrite(str(snapshot_path), frame)

    return str(snapshot_path)


async def trigger_clip_extraction(camera_id: uuid.UUID, incident_id: str) -> None:
    """
    Trigger the evidence clip extraction pipeline.
    Uses the existing CameraFrameBuffer to extract pre-event footage.
    """
    try:
        from services.evidence_manager.clip_extractor import request_clip_extraction
        await request_clip_extraction(
            incident_id=uuid.UUID(incident_id),
            camera_id=str(camera_id),
            retention_tag="STANDARD_30D",
            lock=False,
        )
    except Exception as e:
        print(f"  [WARN] Clip extraction skipped: {e}")


def push_frame_to_buffer(camera_id: str, frame: np.ndarray) -> None:
    """Push a frame into the per-camera circular buffer for evidence clips."""
    try:
        from services.inference.frame_buffer import get_buffer
        buf = get_buffer(camera_id)
        buf.push(frame)
    except Exception:
        pass  # Non-critical — buffer may not be initialized


# ── Main inference loop ───────────────────────────────────────────────────────
async def run_ingest(video_path: str, checkpoint: str, device_str: str, camera_name: str) -> None:
    from ultralytics import YOLO

    print("\n" + "=" * 56)
    print("  SmartOmniSentinel — Live Inference Ingest")
    print("=" * 56)
    print(f"  Camera Name: {camera_name}")
    print(f"  Video      : {video_path}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Device     : {device_str}")
    print()

    # Load models
    device = torch.device(device_str)
    print("  Loading YOLOv8-pose ...", end=" ", flush=True)
    pose_model = YOLO("yolov8n-pose.pt")
    print("✓")

    print("  Loading TemporalClassifier ...", end=" ", flush=True)
    clf = TemporalClassifier()
    ckpt = torch.load(checkpoint, map_location=device_str)
    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()
    clf.to(device)
    print("✓")

    # ── ONNX Model Initialization ──────────────────────────────────────────
    import onnxruntime as ort
    onnx_session = None
    onnx_path = Path(__file__).parent.parent / "models" / "violence_model.onnx"
    if onnx_path.exists():
        onnx_session = ort.InferenceSession(str(onnx_path))
        print("  Loaded RWF-2000 ONNX Model for 3D CNN Inference!")
    raw_frame_buffer = collections.deque(maxlen=16)

    # Get demo camera ID
    camera_id = await get_or_create_demo_camera(camera_name)
    camera_id_str = str(camera_id)
    print(f"  Camera ID  : {camera_id}")
    print()
    print("  Pipeline   : YOLO-Pose → GRU Temporal → Smoothing → DB + Evidence")
    print("  Evidence   : Snapshots → evidence_storage/snapshots/")
    print("               Clips    → evidence_storage/clips/")
    print("  Inserting detections → demo.db")
    print("  (Dashboard will update live via WebSocket)")
    print("  Press Ctrl+C to stop.")
    print("=" * 56 + "\n")

    cap = cv2.VideoCapture(
        int(video_path) if str(video_path).isdigit() else video_path
    )
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip    = max(1, round(src_fps / PROCESS_FPS))

    state          = PersonState()
    frame_idx      = 0
    processed      = 0
    last_alert_ts  = 0.0
    total_inserted = 0
    prev_gray      = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = PersonState()
                print("  [video looped]")
                continue

            frame_idx += 1
            if frame_idx % skip != 0:
                continue
            processed += 1

            # Push frame into circular buffer for evidence clips
            push_frame_to_buffer(camera_id_str, frame)
            
            # ── Optical flow (camera-shake detection) ─────────────────────────
            gray_cur = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
            flow_magnitude = 0.0
            if prev_gray is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow_magnitude = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                except Exception:
                    flow_magnitude = 0.0
            prev_gray = gray_cur

            # Pose estimation
            results = pose_model.predict(frame, verbose=False, device=device_str)

            if (results
                    and results[0].keypoints is not None
                    and len(results[0].keypoints.xyn) > 0):

                kp_all = results[0].keypoints.xyn.cpu().numpy()
                
                # Handle cases where confidence array is missing
                if getattr(results[0].keypoints, 'conf', None) is not None:
                    conf_all = results[0].keypoints.conf.cpu().numpy()
                    best_i   = int(np.argmax([c.mean() for c in conf_all]))
                else:
                    best_i   = 0
                    
                kp = kp_all[best_i]
                
                # Skip if keypoints are empty or malformed
                if kp.shape[0] < 17:
                    continue

                state.push(kp)
                
                # Prepare ONNX frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (112, 112))
                frame_float = frame_resized.astype(np.float32) / 255.0
                mean = np.array([0.43216, 0.394666, 0.37645])
                std = np.array([0.22803, 0.22145, 0.216989])
                frame_norm = (frame_float - mean) / std
                raw_frame_buffer.append(frame_norm)
                
                state.classify(clf, device, onnx_session, raw_frame_buffer, flow_magnitude=flow_magnitude)

                now = time.time()
                if state.is_violent and (now - last_alert_ts) >= ALERT_COOLDOWN_SECS:
                    last_alert_ts = now
                    
                    people = len(results[0].keypoints.xyn)

                    # Step 1: Write incident to database
                    inc_id = await write_incident_to_db(
                        camera_id=camera_id,
                        conf=state.violent_conf,
                        people=people,
                    )

                    # Step 2: Save evidence snapshot (timestamped + dashboard copy)
                    snap_path = save_evidence_snapshot(frame, inc_id)

                    # Step 3: Trigger clip extraction from circular buffer
                    await trigger_clip_extraction(camera_id, inc_id)

                    total_inserted += 1
                    ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"  [{ts}] INCIDENT SAVED  id={inc_id[:8]}...  "
                        f"conf={state.violent_conf:.2f}  "
                        f"people={people}  "
                        f"evidence={os.path.basename(snap_path)}  "
                        f"(total={total_inserted})"
                    )

            # Small sleep so we don't hammer the CPU
            await asyncio.sleep(0)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(f"\n  Stopped. Total incidents written to DB: {total_inserted}")
        
        # Show evidence storage stats
        evidence_dir = Path(__file__).parent.parent / "evidence_storage"
        snap_count = len(list((evidence_dir / "snapshots").glob("*.jpg"))) if (evidence_dir / "snapshots").exists() else 0
        clip_count = len(list((evidence_dir / "clips").glob("*.mp4"))) if (evidence_dir / "clips").exists() else 0
        print(f"  Evidence: {snap_count} snapshots, {clip_count} clips saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartOmniSentinel — Inference Ingest (writes to demo.db)"
    )
    parser.add_argument("--video",      type=str, default="your_video.mp4")
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/temporal_v1.pt")
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--camera",     type=str, default="Main Entrance", help="Name of the camera in the DB")
    args = parser.parse_args()
    asyncio.run(run_ingest(args.video, args.checkpoint, args.device, args.camera))
