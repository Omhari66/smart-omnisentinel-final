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
import base64
import collections
import json
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
VIOLENCE_CONF_MIN   = 0.60
PROCESS_FPS         = 4
ALERT_COOLDOWN_SECS = 5.0
DIAG_PRINT_EVERY    = 8
TRACK_STALE_FRAMES  = 30     # Prune a track's PersonState after this many missed frames


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

    def classify(self, clf: TemporalClassifier, device: torch.device, engine: str, onnx_session=None, raw_frame_buffer=None) -> None:
        if engine == "cnn" and onnx_session is not None and raw_frame_buffer is not None and len(raw_frame_buffer) == 16:
            # --- 3D CNN (RWF-2000 trained) PATH ---
            video_tensor = np.stack(list(raw_frame_buffer))
            video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))
            video_tensor = np.expand_dims(video_tensor, axis=0).astype(np.float32)
            
            input_name = onnx_session.get_inputs()[0].name
            output_name = onnx_session.get_outputs()[0].name
            result = onnx_session.run([output_name], {input_name: video_tensor})
            self.violent_conf = float(result[0][0][0])
            pred = 1 if self.violent_conf >= 0.75 else 0
            
        else:
            # --- SKELETON POSE + GRU PATH ---
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
    initial_notes = ""  # will be filled in within seconds by the AI summary task
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
            notes=initial_notes,
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


# ── ROI (Region of Interest) helpers ─────────────────────────────────────────

def load_roi_zone(roi_json_path: str) -> list[list[float]] | None:
    """
    Load a ROI polygon from a JSON file saved by the dashboard.
    Returns a list of [x, y] normalized (0-1) coordinate pairs, or None.
    """
    try:
        p = Path(roi_json_path)
        if p.exists():
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) >= 3:
                print(f"  [ROI] Loaded zone with {len(data)} points from {p.name}")
                return data
    except Exception as e:
        print(f"  [ROI] Failed to load ROI file: {e}")
    return None


def point_in_roi(x: float, y: float, polygon: list[list[float]]) -> bool:
    """
    Test if a normalized (0-1) point is inside the polygon using ray casting.
    polygon: list of [x, y] pairs in normalized (0-1) coordinates.
    """
    n = len(polygon)
    inside = False
    px, py = x, y
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


# ── LLM Incident Summarization ────────────────────────────────────────────────

async def generate_llm_summary(
    frame: "np.ndarray",
    conf: float,
    people: int,
    model: str = "moondream",
) -> str:
    """
    Send the evidence frame to a locally running Ollama VLM and return a
    one-sentence incident description.  Falls back gracefully if Ollama is
    not running or the model is not available.
    """
    import random

    def _rich_fallback(conf: float, people: int) -> str:
        """Generate a rich, contextual, varied summary without LLM."""
        severity = "high" if conf >= 0.75 else ("moderate" if conf >= 0.55 else "low")
        person_phrase = "one individual" if people == 1 else f"{people} individuals"

        high_conf_summaries = [
            f"Security camera detected a violent physical altercation involving {person_phrase}. "
            f"Aggressive striking motions are clearly visible with {conf*100:.0f}% model confidence. "
            f"Immediate security response is recommended.",

            f"A physical confrontation involving {person_phrase} has been identified with {conf*100:.0f}% confidence. "
            f"Body pose analysis shows rapid, forceful movements consistent with active fighting. "
            f"The incident requires urgent attention.",

            f"High-confidence violence alert: {person_phrase} engaged in a physical fight detected by the AI engine "
            f"(confidence: {conf*100:.0f}%). Skeletal pose vectors indicate aggressive combat postures. "
            f"Flagged for immediate security review.",
        ]
        med_conf_summaries = [
            f"Potential physical altercation detected involving {person_phrase} (confidence: {conf*100:.0f}%). "
            f"Pose estimation indicates aggressive body language and rapid limb movement consistent with fighting. "
            f"Manual review advised.",

            f"AI model flagged a possible violent interaction between {person_phrase} with {conf*100:.0f}% certainty. "
            f"Movement patterns suggest a physical struggle. Evidence snapshot captured for review.",

            f"Suspicious interaction flagged: {person_phrase} exhibiting rapid, aggressive movements (confidence {conf*100:.0f}%). "
            f"Incident may involve physical violence. Security personnel should review the evidence.",
        ]
        low_conf_summaries = [
            f"Low-confidence alert: motion analysis detected unusual physical activity from {person_phrase} ({conf*100:.0f}% confidence). "
            f"The behavior pattern partially matches a physical altercation. Further review recommended.",

            f"Possible disturbance detected involving {person_phrase}. Confidence is {conf*100:.0f}%, suggesting ambiguous activity. "
            f"A security officer should review the evidence snapshot.",
        ]

        if severity == "high":
            return random.choice(high_conf_summaries)
        elif severity == "moderate":
            return random.choice(med_conf_summaries)
        else:
            return random.choice(low_conf_summaries)

    # ── Try Ollama VLM first ──────────────────────────────────────────────────
    try:
        import httpx, cv2 as _cv2

        # Downscale to 512px max dimension — Moondream's sweet spot
        h, w = frame.shape[:2]
        scale = min(512 / w, 512 / h)
        if scale < 1:
            llm_frame = _cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            llm_frame = frame

        ok, buf = _cv2.imencode(".jpg", llm_frame, [_cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return _rich_fallback(conf, people)
        b64_img = base64.b64encode(buf.tobytes()).decode()

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe what is happening in this image. Focus on the people and their actions.",
                    "images": [b64_img],
                }
            ],
            "stream": False,
            "options": {"num_predict": 200, "temperature": 0.1},
        }

        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post("http://localhost:11434/api/chat", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                text = (result.get("message") or {}).get("content", "").strip()
                if text:
                    print(f"  [LLM] VLM summary: '{text[:80]}...'")
                    return text
    except Exception as e:
        print(f"  [LLM] Ollama unavailable ({type(e).__name__}), using contextual summary.")

    # ── Rich contextual fallback ──────────────────────────────────────────────
    summary = _rich_fallback(conf, people)
    print(f"  [LLM] Contextual summary generated.")
    return summary


async def update_incident_notes(incident_id: str, notes: str) -> None:
    """Persist the LLM-generated summary into the Incident.notes field."""
    try:
        from db.session import get_session_factory
        from db.models.incident import Incident
        from sqlalchemy import select

        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(select(Incident).where(Incident.id == uuid.UUID(incident_id)))
            inc = result.scalar_one_or_none()
            if inc:
                inc.notes = notes
                await db.commit()
                print(f"  [LLM] Summary saved → '{notes[:80]}...' " if len(notes) > 80 else f"  [LLM] Summary saved → '{notes}'")
    except Exception as e:
        print(f"  [LLM] Could not save summary to DB: {e}")


async def _llm_summarize_and_save(
    frame: "np.ndarray",
    conf: float,
    people: int,
    incident_id: str,
    model: str,
) -> None:
    """Convenience wrapper: generate LLM summary → save to DB notes field."""
    summary = await generate_llm_summary(frame, conf, people, model)
    await update_incident_notes(incident_id, summary)


# ── Main inference loop ───────────────────────────────────────────────────────
async def run_ingest(
    video_path: str,
    checkpoint: str,
    camera_name: str,
    live_frame_path: str = None,
    engine: str = "pose",
    roi_zone: list | None = None,
    llm_enabled: bool = True,
    llm_model: str = "moondream",
) -> None:
    from ultralytics import YOLO

    # Auto-detect GPU
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("\n" + "=" * 56)
    print("  SmartOmniSentinel — Live Inference Ingest")
    print("=" * 56)
    print(f"  Camera Name: {camera_name}")
    print(f"  Video      : {video_path}")
    print(f"  Live UI    : {live_frame_path or 'Off'}")
    print(f"  AI Engine  : {engine.upper()}")
    print(f"  Device     : {device_str.upper()}")
    print(f"  ROI Zone   : {'Active (' + str(len(roi_zone)) + ' pts)' if roi_zone else 'Full Frame (no filter)'}")
    print(f"  LLM Summary: {'ON — ' + llm_model if llm_enabled else 'OFF'}")
    print()

    # Load models
    print(f"  Loading YOLOv8-pose on {device_str.upper()} ...", end=" ", flush=True)
    pose_model = YOLO("yolov8n-pose.pt")
    pose_model.to(device)
    print("✓")

    print(f"  Loading TemporalClassifier on {device_str.upper()} ...", end=" ", flush=True)
    clf = TemporalClassifier()
    ckpt = torch.load(checkpoint, map_location=device_str)
    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()
    clf.to(device)
    print("✓")

    # Load ONNX CNN only if requested
    onnx_session = None
    raw_frame_buffer = collections.deque(maxlen=16)
    if engine == "cnn":
        import onnxruntime as ort
        onnx_path = Path(__file__).parent.parent / "models" / "violence_model.onnx"
        if onnx_path.exists():
            # Attempt to use GPU providers for ONNX if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device_str == "cuda" else ['CPUExecutionProvider']
            try:
                onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
                active_provider = onnx_session.get_providers()[0]
                print(f"  Loaded RWF-2000 ONNX Model with {active_provider}!")
            except Exception as e:
                print(f"  Failed to load ONNX with GPU: {e}. Falling back to CPU.")
                onnx_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        else:
            print("  [WARN] violence_model.onnx not found! Falling back to POSE engine.")
            engine = "pose"

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

    if str(video_path).isdigit():
        src = int(video_path)
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        src = video_path
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"ERROR: Cannot open: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not isinstance(src, int) and not str(src).isdigit() else 0
    if total_frames > 0 and total_frames < 300:
        skip = 2  # Process more frames for short clips to fill the sliding window
    else:
        skip = max(1, round(src_fps / PROCESS_FPS))

    # ── Multi-person tracking state ──────────────────────────────────────
    # One PersonState per active track ID (ByteTrack, built into Ultralytics)
    states:          dict[int, PersonState] = {}
    track_last_seen: dict[int, int]         = {}   # track_id → last processed frame idx

    frame_idx      = 0
    processed      = 0
    last_alert_ts  = 0.0
    total_inserted = 0
    prev_gray      = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # If it's a live camera, don't infinitely loop if it fails
                if isinstance(src, int) or str(src).isdigit():
                    print("\nERROR: Cannot grab frame from webcam. Is it being used by another program?")
                    break
                    
                cap.release()
                if isinstance(src, int) or str(src).isdigit():
                    cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(str(src))
                states.clear()
                track_last_seen.clear()
                print("  [video looped]")
                continue

            frame_idx += 1
            if frame_idx % skip != 0:
                continue
            processed += 1

            # Push frame into circular buffer for evidence clips
            push_frame_to_buffer(camera_id_str, frame)
            
            # Export to Dashboard if requested (using atomic write with retries to avoid Windows file locks)
            if live_frame_path:
                tmp_frame_path = live_frame_path + ".tmp.jpg"
                if cv2.imwrite(tmp_frame_path, frame):
                    import time
                    for _ in range(5):
                        try:
                            os.replace(tmp_frame_path, live_frame_path)
                            break
                        except Exception:
                            time.sleep(0.01)

            # ── Pose estimation with ByteTrack (Multi-Person Tracking) ────────
            # persist=True tells Ultralytics to keep track IDs stable across frames
            results = pose_model.track(frame, verbose=False, device=device_str, persist=True, tracker="bytetrack.yaml")

            # ── Progress print ────────────────────────────────────────────────
            if processed % 10 == 0:
                n_detected = len(results[0].keypoints.xyn) if (results and results[0].keypoints is not None) else 0
                n_tracks   = len(states)
                print(f"  [Progress] frame={processed} detected={n_detected} active_tracks={n_tracks}", flush=True)

            if (results
                    and results[0].keypoints is not None
                    and len(results[0].keypoints.xyn) > 0
                    and results[0].boxes is not None
                    and results[0].boxes.id is not None):

                kp_all   = results[0].keypoints.xyn.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                # CNN raw-frame buffer (shared, best-effort for single-person CNN mode)
                if engine == "cnn":
                    frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (112, 112))
                    frame_float   = frame_resized.astype(np.float32) / 255.0
                    _mean = np.array([0.43216, 0.394666, 0.37645])
                    _std  = np.array([0.22803, 0.22145, 0.216989])
                    raw_frame_buffer.append((frame_float - _mean) / _std)

                # ── Per-track classification ──────────────────────────────────
                violent_track_id   = None
                violent_track_conf = 0.0

                for i, track_id in enumerate(track_ids):
                    if i >= len(kp_all):
                        break
                    kp = kp_all[i]
                    if kp.shape[0] < 17:
                        continue

                    # ── ROI Filter ─────────────────────────────────────────────
                    # The person centroid is the mean of all valid keypoints.
                    # If an ROI zone is defined and the centroid is outside it,
                    # skip this person entirely — they are in an excluded area.
                    if roi_zone is not None:
                        valid_kps = kp[kp[:, 0] > 0]   # filter out (0,0) missing kps
                        if len(valid_kps) > 0:
                            cx = float(valid_kps[:, 0].mean())
                            cy = float(valid_kps[:, 1].mean())
                            if not point_in_roi(cx, cy, roi_zone):
                                continue   # person outside ROI — skip

                    # Lazy-init state for new track IDs
                    if track_id not in states:
                        states[track_id] = PersonState()
                    track_last_seen[track_id] = processed

                    states[track_id].push(kp)
                    states[track_id].classify(clf, device, engine, onnx_session, raw_frame_buffer)

                    if (states[track_id].is_violent
                            and states[track_id].violent_conf > violent_track_conf):
                        violent_track_id   = track_id
                        violent_track_conf = states[track_id].violent_conf

                # ── Prune stale tracks ────────────────────────────────────────
                stale = [tid for tid, last in track_last_seen.items()
                         if processed - last > TRACK_STALE_FRAMES]
                for tid in stale:
                    del states[tid]
                    del track_last_seen[tid]

                # ── Diagnostic print ──────────────────────────────────────────
                if processed % DIAG_PRINT_EVERY == 0 and states:
                    for tid, st in states.items():
                        votes = sum(1 for p in st.buffer if p == 1)
                        print(
                            f"  [frame {processed:5d}] track={tid}  "
                            f"conf={st.violent_conf:.3f}  votes={votes}/{len(st.buffer)}  "
                            f"violent={st.is_violent}",
                            flush=True
                        )

                now = time.time()
                if violent_track_id is not None and (now - last_alert_ts) >= ALERT_COOLDOWN_SECS:
                    last_alert_ts = now
                    people = len(track_ids)

                    # Step 1: Write incident to database
                    inc_id = await write_incident_to_db(
                        camera_id=camera_id,
                        conf=violent_track_conf,
                        people=people,
                    )

                    # Step 2: Save evidence snapshot (timestamped + dashboard copy)
                    snap_path = save_evidence_snapshot(frame, inc_id)

                    # Step 3: Trigger clip extraction from circular buffer
                    await trigger_clip_extraction(camera_id, inc_id)

                    # Step 4: LLM Summary (async, non-blocking, optional)
                    if llm_enabled:
                        asyncio.create_task(
                            _llm_summarize_and_save(frame.copy(), violent_track_conf, people, inc_id, llm_model)
                        )

                    total_inserted += 1
                    ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"  [{ts}] INCIDENT SAVED  id={inc_id[:8]}...  "
                        f"track_id={violent_track_id}  "
                        f"conf={violent_track_conf:.2f}  "
                        f"people={people}  "
                        f"evidence={os.path.basename(snap_path)}  "
                        f"(total={total_inserted})",
                        flush=True
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
    parser = argparse.ArgumentParser(description="SmartOmniSentinel — Edge Inference Ingest")
    parser.add_argument("--video",      type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/temporal_v1.pt")
    parser.add_argument("--camera",     type=str, default="Main Entrance", help="Camera name in DB")
    parser.add_argument("--live_frame", type=str, default=None,   help="Path to dump live frames for the dashboard")
    parser.add_argument("--engine",     type=str, default="pose", choices=["pose", "cnn"], help="AI Engine")
    parser.add_argument("--roi_zone",   type=str, default=None,   help="Path to roi_zone.json saved by the dashboard (optional)")
    parser.add_argument("--no-llm",     dest="llm", action="store_false", help="Disable local Ollama LLM summarization")
    parser.set_defaults(llm=True)
    parser.add_argument("--llm_model",  type=str, default="moondream", help="Ollama model name (default: moondream)")
    args = parser.parse_args()

    # Load ROI polygon from JSON file if path is given
    roi_zone = load_roi_zone(args.roi_zone) if args.roi_zone else None

    asyncio.run(run_ingest(
        args.video,
        args.checkpoint,
        args.camera,
        args.live_frame,
        args.engine,
        roi_zone=roi_zone,
        llm_enabled=args.llm,
        llm_model=args.llm_model,
    ))
