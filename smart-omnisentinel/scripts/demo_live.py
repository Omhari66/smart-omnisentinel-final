"""
scripts/demo_live.py
---------------------
SmartOmniSentinel — Standalone Visual Demo

Runs pose estimation + temporal GRU classifier on a video file,
displaying real-time overlays: skeleton, bounding box, confidence,
temporal smoothing buffer, and VIOLENCE DETECTED alert banner.

Does NOT require the full API stack — runs in isolation.

Usage:
    python scripts/demo_live.py --video your_video.mp4
    python scripts/demo_live.py --video 0               # webcam
    python scripts/demo_live.py --video your_video.mp4 --save  # save output

Controls:
    q = quit   |   p = pause/resume   |   s = screenshot
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.models.temporal_model import TemporalClassifier

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE          = 16
FEATURE_DIM          = 34      # 17 joints × (x, y) — must match training
SMOOTHING_WINDOW     = 8       # rolling prediction buffer length
SMOOTHING_THRESHOLD  = 5       # violent frames in buffer to trigger alert
VIOLENCE_CONF_MIN    = 0.75    # raised to 0.75 for RWF-2000 3D CNN (was 0.60)
PROCESS_FPS          = 4       # effective frame rate for inference
ALERT_COOLDOWN_SECS  = 3.0    # minimum seconds between consecutive printed alerts
CAMERA_SHAKE_THRESH  = 12.0   # global optical flow above this = camera moving, not violence

# COCO-17 skeleton pairs (joint index → joint index)
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # head
    (5,6),                             # shoulders
    (5,7),(7,9),(6,8),(8,10),          # arms
    (5,11),(6,12),(11,12),             # torso
    (11,13),(13,15),(12,14),(14,16),   # legs
]

# Colours (BGR)
C_GREEN  = (50, 220, 50)
C_RED    = (30,  30, 220)
C_ORANGE = (30, 160, 255)
C_WHITE  = (255, 255, 255)
C_DARK   = (20,  20,  20)
C_CYAN   = (220, 200, 40)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models(checkpoint: str, device: str):
    print("  Loading YOLOv8-pose ...", end=" ", flush=True)
    from ultralytics import YOLO
    pose_model = YOLO(str(ROOT / "yolov8n-pose.pt"))
    print("✓")

    print("  Loading TemporalClassifier ...", end=" ", flush=True)
    clf = TemporalClassifier()
    ckpt = torch.load(checkpoint, map_location=device)
    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()
    clf.to(device)
    best_f1 = ckpt.get("best_val_f1", "N/A")
    print(f"✓  (checkpoint val F1 = {best_f1:.4f})" if isinstance(best_f1, float) else "✓")

    return pose_model, clf


# ── Per-person sliding window + smoother ────────────────────────────────────
class PersonState:
    def __init__(self):
        self.feature_window: collections.deque = collections.deque(maxlen=WINDOW_SIZE)
        self.pred_buffer:    collections.deque = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.violent_conf:   float = 0.0
        self.is_violent:     bool  = False

    def push(self, keypoints: np.ndarray) -> None:
        """keypoints: (17, 2) normalised"""
        flat = keypoints.flatten().astype(np.float32)
        if flat.shape[0] == FEATURE_DIM:
            self.feature_window.append(flat)

    def classify(self, clf: TemporalClassifier, device: torch.device, onnx_session=None, raw_frame_buffer=None, flow_magnitude: float = 0.0) -> None:
        # --- Camera-shake guard: if whole frame is shaking, skip detection ---
        if flow_magnitude > CAMERA_SHAKE_THRESH:
            # Reset violent state — this is camera movement, not violence
            self.pred_buffer.append(0)
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
            if len(self.feature_window) < WINDOW_SIZE:
                return
            x = torch.tensor(
                np.array(list(self.feature_window)), dtype=torch.float32
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(clf(x), dim=-1)[0].cpu().numpy()

            pred = int(np.argmax(probs))
            self.violent_conf = float(probs[1])

        self.pred_buffer.append(pred)
        violent_votes = sum(1 for p in self.pred_buffer if p == 1)
        self.is_violent = (
            violent_votes >= SMOOTHING_THRESHOLD
            and self.violent_conf >= VIOLENCE_CONF_MIN
        )

    @property
    def buffer_fill(self) -> int:
        return len(self.feature_window)

    @property
    def violent_votes(self) -> int:
        return sum(1 for p in self.pred_buffer if p == 1)


# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_skeleton(
    frame: np.ndarray,
    kp: np.ndarray,       # (17, 2) normalised
    conf: np.ndarray,     # (17,)
    color: tuple,
    conf_thresh: float = 0.4,
) -> None:
    h, w = frame.shape[:2]
    for i, j in SKELETON:
        if conf[i] >= conf_thresh and conf[j] >= conf_thresh:
            p1 = (int(kp[i, 0] * w), int(kp[i, 1] * h))
            p2 = (int(kp[j, 0] * w), int(kp[j, 1] * h))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(kp):
        if conf[i] >= conf_thresh:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, color, -1, cv2.LINE_AA)


def draw_bbox(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple,
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 2, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    state: PersonState | None,
    frame_idx: int,
    processed: int,
    fps: float,
    alert_flash: float,
    event_log: list[str],
    onnx_session=None,
) -> None:
    h, w = frame.shape[:2]
    is_violent = state.is_violent if state else False

    # ── Top banner ────────────────────────────────────────────────────────
    banner_h = 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    title = "SmartOmniSentinel  |  AI Violence Detection"
    cv2.putText(frame, title, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_CYAN, 2, cv2.LINE_AA)

    status_txt = f"F:{frame_idx:05d}  P:{processed:04d}  {fps:.1f}fps"
    (sw, _), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, status_txt, (w - sw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # ── VIOLENCE ALERT flash overlay ─────────────────────────────────────
    now = time.time()
    if is_violent or (now - alert_flash < 1.5):
        alpha = 0.35 if is_violent else max(0.0, 0.35 * (1.5 - (now - alert_flash)) / 1.5)
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), C_RED, -1)
        cv2.addWeighted(flash, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "! VIOLENCE DETECTED !", (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, C_WHITE, 3, cv2.LINE_AA)

    # ── Temporal smoothing bar (bottom-left) ──────────────────────────────
    bar_x, bar_y = 12, h - 50
    cv2.putText(frame, "Temporal buffer:", (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    cell = 18
    votes = state.violent_votes if state else 0
    total_buf = len(state.pred_buffer) if state else 0
    for i in range(SMOOTHING_WINDOW):
        cx = bar_x + i * (cell + 2)
        cy = bar_y
        if i < total_buf:
            buf_list = list(state.pred_buffer) if state else []
            filled_color = C_RED if (i < len(buf_list) and buf_list[i] == 1) else C_GREEN
        else:
            filled_color = (60, 60, 60)
        cv2.rectangle(frame, (cx, cy), (cx + cell, cy + cell), filled_color, -1)
        cv2.rectangle(frame, (cx, cy), (cx + cell, cy + cell), (100, 100, 100), 1)

    thresh_x = bar_x + SMOOTHING_THRESHOLD * (cell + 2) - 2
    cv2.line(frame, (thresh_x, bar_y - 4), (thresh_x, bar_y + cell + 4), C_ORANGE, 2)
    cv2.putText(frame, "threshold", (thresh_x - 30, bar_y + cell + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_ORANGE, 1, cv2.LINE_AA)

    # ── Confidence bar + model source label ───────────────────────────────
    if state and (state.buffer_fill >= WINDOW_SIZE or onnx_session is not None):
        conf_val = state.violent_conf
        bar_w = 200
        bx = w - bar_w - 14
        by = h - 50
        model_src = "RWF-CNN" if onnx_session is not None else "Skeleton"
        cv2.putText(frame, f"Violence conf [{model_src}]: {conf_val:.0%}", (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 14), (60, 60, 60), -1)
        fill = int(conf_val * bar_w)
        fill_color = C_RED if conf_val >= VIOLENCE_CONF_MIN else C_GREEN
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 14), fill_color, -1)
        thresh_px = bx + int(VIOLENCE_CONF_MIN * bar_w)
        cv2.line(frame, (thresh_px, by - 2), (thresh_px, by + 16), C_ORANGE, 2)

    # ── Event log (bottom-right) ──────────────────────────────────────────
    for i, ev in enumerate(reversed(event_log[-4:])):
        alpha_text = 1.0 - i * 0.22
        color_ev = tuple(int(c * alpha_text) for c in C_CYAN)
        cv2.putText(frame, ev, (w // 2, h - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color_ev, 1, cv2.LINE_AA)


# ── Main demo loop ────────────────────────────────────────────────────────────
def run_demo(
    video_path: str,
    checkpoint: str,
    device_str: str = "cpu",
    save_output: bool = False,
) -> None:
    print("\n" + "=" * 56)
    print("  SmartOmniSentinel — Violence Detection Demo")
    print("=" * 56)
    print(f"  Source     : {video_path}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Device     : {device_str}")
    print()

    device = torch.device(device_str)
    pose_model, clf = load_models(checkpoint, device_str)

    # Open video
    if str(video_path).isdigit():
        src = int(video_path)
        # Use DirectShow backend on Windows for webcams, fixes MSMF grabFrame errors
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        src = video_path
        cap = cv2.VideoCapture(src)
        
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {video_path}")
        sys.exit(1)

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip      = max(1, round(src_fps / PROCESS_FPS))

    print(f"  Video      : {vid_w}×{vid_h} @ {src_fps:.1f}fps")
    print(f"  Processing : every {skip}th frame (~{PROCESS_FPS}fps)")
    print()
    print("  Controls: Q=quit  P=pause  S=screenshot")
    print("=" * 56 + "\n")

    writer = None
    if save_output:
        out_path = str(ROOT / "scripts" / "output" / "demo_output.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"),
            PROCESS_FPS, (vid_w, vid_h)
        )
        print(f"  Saving to : {out_path}\n")

    state      = PersonState()
    frame_idx  = 0
    processed  = 0
    t_start    = time.time()
    alert_flash_ts = 0.0
    event_log: list[str] = []
    paused     = False
    prev_gray  = None          # for optical flow camera-shake detection

    # ── ONNX Model Initialization ──────────────────────────────────────────
    import onnxruntime as ort
    onnx_session = None
    onnx_path = ROOT / "models" / "violence_model.onnx"
    if onnx_path.exists():
        onnx_session = ort.InferenceSession(str(onnx_path))
        print("  Loaded RWF-2000 ONNX Model for 3D CNN Inference!")
    
    raw_frame_buffer = collections.deque(maxlen=16)

    cv2.namedWindow("SmartOmniSentinel", cv2.WINDOW_NORMAL)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
            print("  [PAUSED]" if paused else "  [RESUMED]")
        if key == ord('s'):
            ss_path = str(ROOT / "scripts" / "output" / f"screenshot_{int(time.time())}.jpg")
            os.makedirs(os.path.dirname(ss_path), exist_ok=True)
            cv2.imwrite(ss_path, frame if 'frame' in dir() else np.zeros((480, 640, 3), dtype=np.uint8))
            print(f"  Screenshot saved: {ss_path}")

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            # If it's a live camera (video source is integer) and it fails, it's likely locked
            if isinstance(src, int) or str(src).isdigit():
                print("\nERROR: Cannot grab frame from webcam. Is it being used by another program?")
                break
                
            # Loop video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            state = PersonState()
            continue

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        processed += 1
        elapsed = time.time() - t_start
        live_fps = processed / elapsed if elapsed > 0 else 0.0

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

        # ── Pose estimation ───────────────────────────────────────────────
        results = pose_model.predict(frame, verbose=False, device=device_str)

        if (results
                and results[0].keypoints is not None
                and len(results[0].keypoints.xyn) > 0):

            kp_all   = results[0].keypoints.xyn.cpu().numpy()   # (N, 17, 2)
            boxes    = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else None

            # Handle cases where confidence array is missing
            if getattr(results[0].keypoints, 'conf', None) is not None:
                conf_all = results[0].keypoints.conf.cpu().numpy()
                best_i = int(np.argmax([c.mean() for c in conf_all]))
                conf = conf_all[best_i]
            else:
                best_i = 0
                conf = np.ones(17, dtype=np.float32)

            kp   = kp_all[best_i]    # (17, 2)
            
            # Skip if keypoints are empty or malformed
            if kp.shape[0] < 17:
                continue

            # Feed into sliding window
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

            # Bounding box
            if boxes is not None and best_i < len(boxes):
                x1, y1, x2, y2 = map(int, boxes[best_i])
                skel_color = C_RED if state.is_violent else C_GREEN
                label = (
                    f"VIOLENT  {state.violent_conf:.0%}"
                    if state.is_violent else
                    f"Normal  {1 - state.violent_conf:.0%}"
                )
                draw_bbox(frame, x1, y1, x2, y2, label, skel_color)

            skel_color = C_RED if state.is_violent else C_GREEN
            draw_skeleton(frame, kp, conf, skel_color)

            # Alert trigger — cooldown prevents spam
            if state.is_violent:
                now = time.time()
                if now - alert_flash_ts >= ALERT_COOLDOWN_SECS:
                    ts = time.strftime("%H:%M:%S")
                    msg = f"[{ts}] ALERT  conf={state.violent_conf:.2f}"
                    event_log.append(msg)
                    alert_flash_ts = now
                    print(f"  {msg}")

        else:
            # No person detected — show idle state
            cv2.putText(frame, "No person detected", (12, vid_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # ── Draw HUD ──────────────────────────────────────────────────────
        draw_hud(frame, state, frame_idx, processed, live_fps,
                 alert_flash_ts, event_log, onnx_session=onnx_session)

        cv2.imshow("SmartOmniSentinel", frame)
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n  Processed {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f}fps)")
    print(f"  Total violence alerts fired: {len(event_log)}")
    if event_log:
        print("\n  Alert log:")
        for ev in event_log:
            print(f"    {ev}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartOmniSentinel — Live Visual Demo"
    )
    parser.add_argument(
        "--video", type=str, default="your_video.mp4",
        help="Path to video file, or webcam index (e.g. 0)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="ml/checkpoints/temporal_v1.pt",
        help="Path to trained TemporalClassifier checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device: cpu | cuda"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output video to scripts/output/demo_output.mp4"
    )
    args = parser.parse_args()
    run_demo(args.video, args.checkpoint, args.device, args.save)
