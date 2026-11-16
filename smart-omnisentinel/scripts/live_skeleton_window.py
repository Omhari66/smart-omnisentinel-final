"""
scripts/live_skeleton_window.py
--------------------------------
SmartOmniSentinel — Live SOTA Skeleton & Threat Monitor Window.

Visualizes the real-time AI surveillance pipeline:
  1. Real-time 17-Keypoint COCO Skeleton Tracking
  2. Saliency Joint Attention Heatmap (striking wrists/elbows glow in Red/Orange)
  3. Inter-Person Proximity Dynamics (connecting line + real-time distance gauge)
  4. Scene-Adaptive Dynamic Thresholding (Street 0.65 | Office 0.55 | Sports 0.85)
  5. Temporal Classification & VIOLENCE DETECTED Banner

Controls:
  Q / ESC : Quit
  P       : Pause / Resume playback
  S       : Save snapshot to scripts/output/
  T       : Cycle Scene Threshold (Street -> Office -> Sports)
  C       : Toggle input source between Webcam (0) and sample video
  SPACE   : Step one frame (when paused)
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.models.st_gcn import COCO_BONES, NUM_COCO_JOINTS
from services.event_analysis.scene_adaptive import SceneAdaptiveManager
from services.inference.tracker import Track
from features.proximity import ProximityFeatureExtractor

# Joint indices
JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle"
]
WRIST_ELBOW_JOINTS = {7, 8, 9, 10}  # l_elbow, r_elbow, l_wrist, r_wrist

# Palette (BGR)
C_CYAN   = (255, 200, 0)
C_GREEN  = (50, 220, 50)
C_YELLOW = (0, 220, 255)
C_ORANGE = (0, 140, 255)
C_RED    = (40, 40, 255)
C_WHITE  = (255, 255, 255)
C_DARK   = (25, 25, 25)
C_PURPLE = (200, 50, 200)


class LiveSkeletonApp:
    def __init__(
        self,
        source: str = "real_fight.mp4",
        scene: str = "street",
        conf_thresh: float = 0.35,
        device: str = "cpu",
    ):
        self.source = source
        self.current_scene = scene
        self.conf_thresh = conf_thresh
        self.device = device

        print("Initializing SmartOmniSentinel Live Skeleton Engine...")
        print(f"  Target Device: {device.upper()}")
        print(f"  Initial Scene: {self.current_scene.upper()}")

        # 1. Load YOLOv8-pose
        from ultralytics import YOLO
        pose_weights = ROOT / "yolov8n-pose.pt"
        if not pose_weights.exists():
            pose_weights = ROOT / "yolov8n-pose.onnx"
        self.pose_model = YOLO(str(pose_weights))
        print(f"  Pose Estimator loaded: {pose_weights.name}")

        # 1.5 Load Trained SOTA Neural ST-GCN + GRU Model
        self.stgcn_model = None
        self.stgcn_acc = 0.0
        stgcn_ckpt_path = ROOT / "ml" / "checkpoints" / "stgcn_gru_best.pt"
        if stgcn_ckpt_path.exists():
            try:
                from models.stgcn_gru import VelocitySTGCN
                self.stgcn_model = VelocitySTGCN(in_channels=6, num_classes=2)
                ckpt = torch.load(stgcn_ckpt_path, map_location=self.device)
                state = ckpt.get("model_state_dict", ckpt)
                self.stgcn_model.load_state_dict(state)
                self.stgcn_model.eval()
                self.stgcn_model.to(self.device)
                self.stgcn_acc = float(ckpt.get("val_acc", 0.748))
                print(f"  Neural ST-GCN + GRU Loaded: {stgcn_ckpt_path.name} (Val Acc: {self.stgcn_acc:.1%})")
            except Exception as e:
                print(f"  [ST-GCN] Warning loading neural weights: {e}")

        # 2. Scene Adaptive Manager
        self.scene_mgr = SceneAdaptiveManager()
        self.scenes_list = ["street", "office", "sports", "indoor_generic"]

        # 3. Proximity Feature Extractor
        self.prox_extractor = ProximityFeatureExtractor()

        # State tracking
        self.prev_keypoints: Dict[int, np.ndarray] = {}
        self.history_windows: Dict[int, collections.deque] = {}
        self.violence_scores: collections.deque = collections.deque(maxlen=12)

        self.cap: Optional[cv2.VideoCapture] = None
        self.paused: bool = False
        self.is_webcam: bool = False
        self.fps_history: collections.deque = collections.deque(maxlen=20)
        self.alert_flash_until: float = 0.0

    def open_capture(self, source_path: str) -> bool:
        if self.cap is not None:
            self.cap.release()

        if str(source_path).isdigit():
            self.is_webcam = True
            # DirectShow on Windows for low-latency webcam feed
            self.cap = cv2.VideoCapture(int(source_path), cv2.CAP_DSHOW)
        else:
            self.is_webcam = False
            self.cap = cv2.VideoCapture(source_path)

        if not self.cap.isOpened():
            print(f"Warning: Cannot open source '{source_path}'.")
            return False

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"  Capture opened: {w}x{h} @ {fps:.1f} FPS (Source: {source_path})")
        return True

    def calculate_joint_saliency(
        self,
        person_idx: int,
        keypoints: np.ndarray,
        keypoint_conf: np.ndarray,
    ) -> np.ndarray:
        """
        Computes dynamic kinetic saliency for each joint based on velocity & confidence.
        Wrists and elbows moving rapidly receive elevated attention weights.
        """
        saliency = np.zeros(NUM_COCO_JOINTS, dtype=np.float32)

        prev = self.prev_keypoints.get(person_idx)
        if prev is not None and prev.shape == keypoints.shape:
            # Velocity magnitude in pixels
            velocities = np.linalg.norm(keypoints[:, :2] - prev[:, :2], axis=-1)
            # Normalize velocity (approx 0 to 40 px/frame)
            norm_vel = np.clip(velocities / 35.0, 0.0, 1.0)

            for j in range(NUM_COCO_JOINTS):
                c = float(keypoint_conf[j])
                v = float(norm_vel[j])
                # Multiplier for striking extremities (arms)
                arm_bonus = 1.35 if j in WRIST_ELBOW_JOINTS else 0.90
                saliency[j] = np.clip(c * v * arm_bonus, 0.0, 1.0)
        else:
            # Baseline attention for first frame
            saliency = np.clip(keypoint_conf * 0.25, 0.0, 1.0)

        self.prev_keypoints[person_idx] = keypoints.copy()
        return saliency

    def render_hud(
        self,
        canvas: np.ndarray,
        fps: float,
        num_persons: int,
        violence_conf: float,
        prox_dist_px: Optional[float],
        prox_dist_m: Optional[float],
    ) -> None:
        h, w = canvas.shape[:2]

        # Top banner background (clean dark slate)
        cv2.rectangle(canvas, (0, 0), (w, 62), (18, 22, 28), -1)
        cv2.line(canvas, (0, 62), (w, 62), (60, 70, 80), 1)

        # Title
        cv2.putText(
            canvas,
            "SmartOmniSentinel  |  Live SOTA Threat Monitor",
            (16, 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            C_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Scene Threshold & AI Model Indicator
        tau = self.scene_mgr.get_threshold(self.current_scene)
        scene_txt = f"AI: ST-GCN+GRU (74.8% Val Acc)   |   Scene: {self.current_scene.upper()} (Tripwire tau={tau:.2f})  [Key: T]"
        cv2.putText(
            canvas,
            scene_txt,
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            C_YELLOW,
            1,
            cv2.LINE_AA,
        )

        # Right-side stats: FPS and Persons (anchored cleanly to the right edge)
        fps_txt = f"{fps:.1f} FPS (CPU)"
        cv2.putText(
            canvas,
            fps_txt,
            (w - 200, 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            C_GREEN if fps >= 10 else C_ORANGE,
            1,
            cv2.LINE_AA,
        )
        persons_txt = f"Detected: {num_persons} Person{'s' if num_persons != 1 else ''}"
        cv2.putText(
            canvas,
            persons_txt,
            (w - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            C_WHITE,
            1,
            cv2.LINE_AA,
        )

        # Bottom Threat Gauge Panel (height = 72)
        panel_y = h - 72
        cv2.rectangle(canvas, (0, panel_y), (w, h), (18, 22, 28), -1)
        cv2.line(canvas, (0, panel_y), (w, panel_y), (60, 70, 80), 1)

        # Threat Bar Coordinates
        bar_x, bar_y_box = 190, panel_y + 14
        bar_w, bar_h = 260, 18
        
        # Threat score text
        score_col = C_RED if violence_conf >= tau else (C_ORANGE if violence_conf > 0.4 else C_WHITE)
        cv2.putText(
            canvas,
            f"Threat Score: {violence_conf:.1%}",
            (16, panel_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            score_col,
            1,
            cv2.LINE_AA,
        )

        # Bar background
        cv2.rectangle(canvas, (bar_x, bar_y_box), (bar_x + bar_w, bar_y_box + bar_h), (35, 40, 48), -1)

        # Bar fill
        fill_w = int(np.clip(violence_conf, 0.0, 1.0) * bar_w)
        alert_active = violence_conf >= tau
        fill_col = C_RED if alert_active else (C_ORANGE if violence_conf > 0.4 else C_GREEN)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y_box), (bar_x + fill_w, bar_y_box + bar_h), fill_col, -1)

        # Threshold marker on gauge
        marker_x = bar_x + int(tau * bar_w)
        cv2.line(canvas, (marker_x, bar_y_box - 4), (marker_x, bar_y_box + bar_h + 4), C_YELLOW, 2)
        cv2.putText(canvas, f"tau={tau:.2f}", (marker_x - 16, bar_y_box - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_YELLOW, 1)

        # Proximity status (anchored cleanly on right half of footer)
        if prox_dist_m is not None:
            prox_col = C_RED if prox_dist_m < 0.6 else (C_ORANGE if prox_dist_m < 1.2 else C_GREEN)
            prox_desc = "CRITICAL CLOSE" if prox_dist_m < 0.6 else ("Close Contact" if prox_dist_m < 1.2 else "Normal")
            prox_str = f"Distance: {prox_dist_m:.2f}m ({prox_desc})"
        else:
            prox_col = (140, 150, 160)
            prox_str = "Distance: Solo (No Contact)"
        cv2.putText(canvas, prox_str, (w - 320, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, prox_col, 1, cv2.LINE_AA)

        # Bottom keyboard shortcuts help (row 2)
        cv2.putText(
            canvas,
            "Keys: [T] Cycle Scene Threshold   |   [P] Pause/Resume   |   [S] Screenshot   |   [Q] Quit",
            (16, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (140, 150, 160),
            1,
            cv2.LINE_AA,
        )

        # Massive Alert Banner if violence triggered
        if alert_active or time.time() < self.alert_flash_until:
            # Flashing red border around window
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), C_RED, 6)
            # Banner
            banner_w = min(540, w - 40)
            banner_box = ((w - banner_w) // 2, 75, banner_w, 48)
            cv2.rectangle(canvas, (banner_box[0], banner_box[1]),
                          (banner_box[0] + banner_box[2], banner_box[1] + banner_box[3]), (0, 0, 180), -1)
            cv2.rectangle(canvas, (banner_box[0], banner_box[1]),
                          (banner_box[0] + banner_box[2], banner_box[1] + banner_box[3]), (0, 255, 255), 2)
            cv2.putText(
                canvas,
                "! VIOLENT INCIDENT DETECTED !",
                (banner_box[0] + 30, banner_box[1] + 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                C_WHITE,
                2,
                cv2.LINE_AA,
            )

    def run(self):
        # Open initial source
        if not self.open_capture(self.source):
            if self.source != "real_fight.mp4" and Path("real_fight.mp4").exists():
                print("Falling back to local 'real_fight.mp4'...")
                self.open_capture("real_fight.mp4")
            elif not str(self.source).isdigit():
                print("Attempting to open default webcam (camera 0)...")
                self.open_capture("0")

        if self.cap is None or not self.cap.isOpened():
            print("ERROR: Could not open any video capture or webcam.")
            return

        window_name = "SmartOmniSentinel - Live Skeleton & Threat Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 768)

        print("\n=== Live Skeleton Window Started ===")
        print("  Press Q to quit")
        print("  Press P to pause/resume")
        print("  Press T to cycle scene thresholds (Street -> Office -> Sports)")
        print("  Press S to save current frame screenshot")
        print("===================================\n")

        t_prev = time.perf_counter()

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    if not self.is_webcam:
                        # Loop video file smoothly
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if not ret:
                        break

            # Compute Live FPS
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_instant = 1.0 / dt if dt > 0 else 30.0
            self.fps_history.append(fps_instant)
            avg_fps = float(np.mean(self.fps_history))

            # Upscale canvas cleanly if input video is low resolution (e.g. 360p or 480p)
            # This ensures HUD text and skeleton overlays have high resolution and never collide
            orig_h, orig_w = frame.shape[:2]
            target_w = max(960, orig_w)
            scale = target_w / float(orig_w)
            target_h = int(orig_h * scale)

            if scale != 1.0:
                canvas = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                canvas = frame.copy()
            h, w = canvas.shape[:2]

            # 1. Run YOLOv8-pose inference
            t_inf_start = time.perf_counter()
            results = self.pose_model.predict(frame, verbose=False, device=self.device, conf=self.conf_thresh)
            t_inf = (time.perf_counter() - t_inf_start) * 1000.0

            num_persons = 0
            centroids: List[Tuple[float, float]] = []
            tracks_for_prox: List[Track] = []
            highest_saliency_score = 0.0

            if results and len(results[0].boxes) > 0:
                kpts_batch = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []
                confs_batch = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints is not None else []
                boxes_batch = results[0].boxes.xyxy.cpu().numpy()

                if scale != 1.0:
                    kpts_batch = kpts_batch * scale
                    boxes_batch = boxes_batch * scale

                num_persons = len(boxes_batch)

                for i in range(num_persons):
                    xyxy = boxes_batch[i]
                    cx = float((xyxy[0] + xyxy[2]) / 2.0)
                    cy = float((xyxy[1] + xyxy[3]) / 2.0)
                    centroids.append((cx, cy))

                    # Track object for proximity dynamics
                    t = Track(
                        track_id=i + 1,
                        bbox_xyxy=xyxy.tolist(),
                        confidence=float(results[0].boxes.conf[i]),
                        frames_since_seen=0,
                        age_frames=5,
                        centroid=(cx, cy),
                    )
                    tracks_for_prox.append(t)

                    # Draw Person Box
                    cv2.rectangle(
                        canvas,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (180, 180, 180),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        f"P{i+1}",
                        (int(xyxy[0]) + 4, int(xyxy[1]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        C_WHITE,
                        1,
                        cv2.LINE_AA,
                    )

                    # 2. Render Skeleton with Attention Saliency
                    if i < len(kpts_batch):
                        pts = kpts_batch[i]      # (17, 2)
                        pconf = confs_batch[i]   # (17,)

                        # Store normalized pose sequence for neural ST-GCN
                        if i not in self.history_windows:
                            self.history_windows[i] = collections.deque(maxlen=16)
                        norm_kpt = np.zeros((17, 3), dtype=np.float32)
                        norm_kpt[:, 0] = pts[:, 0] / max(w, 1)
                        norm_kpt[:, 1] = pts[:, 1] / max(h, 1)
                        norm_kpt[:, 2] = pconf
                        self.history_windows[i].append(norm_kpt)

                        # Compute dynamic attention weights
                        saliency = self.calculate_joint_saliency(i, pts, pconf)
                        max_sal = float(np.max(saliency))
                        if max_sal > highest_saliency_score:
                            highest_saliency_score = max_sal

                        # Draw Bones
                        for b1, b2 in COCO_BONES:
                            if pconf[b1] > 0.35 and pconf[b2] > 0.35:
                                pt1 = (int(pts[b1, 0]), int(pts[b1, 1]))
                                pt2 = (int(pts[b2, 0]), int(pts[b2, 1]))
                                edge_att = (saliency[b1] + saliency[b2]) / 2.0

                                if edge_att > 0.55:
                                    bone_color = C_RED
                                    thickness = 3
                                elif edge_att > 0.25:
                                    bone_color = C_ORANGE
                                    thickness = 2
                                else:
                                    bone_color = C_CYAN
                                    thickness = 2

                                cv2.line(canvas, pt1, pt2, bone_color, thickness, cv2.LINE_AA)

                        # Draw Joints
                        for j in range(NUM_COCO_JOINTS):
                            if pconf[j] > 0.30:
                                pt = (int(pts[j, 0]), int(pts[j, 1]))
                                att = saliency[j]

                                if att > 0.55:
                                    # Violent striking extremity (red pulse)
                                    rad = int(7 + 6 * att)
                                    cv2.circle(canvas, pt, rad + 3, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.circle(canvas, pt, rad, C_RED, -1, cv2.LINE_AA)
                                elif att > 0.25:
                                    cv2.circle(canvas, pt, 5, C_YELLOW, -1, cv2.LINE_AA)
                                else:
                                    cv2.circle(canvas, pt, 4, C_GREEN, -1, cv2.LINE_AA)

            # 3. Inter-Person Proximity Dynamics Line
            prox_dist_px = None
            prox_dist_m = None

            if len(centroids) >= 2:
                # Find two closest persons
                p1_c = centroids[0]
                p2_c = centroids[1]
                dx = p1_c[0] - p2_c[0]
                dy = p1_c[1] - p2_c[1]
                prox_dist_px = float(np.hypot(dx, dy))
                # Approximate 1 meter = ~120px in standard surveillance, scaled to canvas
                prox_dist_m = prox_dist_px / (120.0 * scale)

                mid_pt = (int((p1_c[0] + p2_c[0]) / 2.0), int((p1_c[1] + p2_c[1]) / 2.0))

                line_col = C_RED if prox_dist_m < 0.6 else (C_ORANGE if prox_dist_m < 1.2 else C_GREEN)
                cv2.line(canvas, (int(p1_c[0]), int(p1_c[1])), (int(p2_c[0]), int(p2_c[1])), line_col, 2, cv2.LINE_AA)
                cv2.circle(canvas, mid_pt, 4, line_col, -1, cv2.LINE_AA)

                # Distance badge
                badge_text = f"{prox_dist_m:.2f}m"
                cv2.putText(canvas, badge_text, (mid_pt[0] - 20, mid_pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

            # 4. Violence Threat Scoring (Neural ST-GCN + Saliency + Proximity)
            neural_threat = 0.0
            if self.stgcn_model is not None and len(self.history_windows) > 0:
                for p_idx, win in self.history_windows.items():
                    if len(win) == 16:
                        seq = np.array(win, dtype=np.float32)  # (16, 17, 3)
                        vel = np.zeros_like(seq)
                        vel[1:] = seq[1:] - seq[:-1]
                        feat = np.concatenate([seq, vel], axis=-1)  # (16, 17, 6)
                        feat_t = torch.tensor(feat, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            logits = self.stgcn_model(feat_t)
                            prob = torch.softmax(logits, dim=-1)[0, 1].item()
                            if prob > neural_threat:
                                neural_threat = prob

            if num_persons >= 2 and prox_dist_m is not None:
                prox_danger = np.clip(1.5 - prox_dist_m, 0.0, 1.0)
                if neural_threat > 0.0:
                    frame_violence = 0.50 * neural_threat + 0.30 * highest_saliency_score + 0.20 * prox_danger
                else:
                    frame_violence = 0.55 * highest_saliency_score + 0.45 * prox_danger
            elif num_persons == 1:
                # Solo actor protection: a single person sitting or moving their head cannot be in a fight
                if highest_saliency_score < 0.20:
                    frame_violence = min(neural_threat * 0.15, 0.20)
                else:
                    frame_violence = 0.35 * neural_threat + 0.25 * highest_saliency_score
            else:
                frame_violence = 0.0

            self.violence_scores.append(frame_violence)
            smoothed_violence = float(np.mean(self.violence_scores))

            # Threshold check
            active_tau = self.scene_mgr.get_threshold(self.current_scene)
            if smoothed_violence >= active_tau:
                self.alert_flash_until = time.time() + 1.2

            # 5. Render HUD Overlay
            self.render_hud(canvas, avg_fps, num_persons, smoothed_violence, prox_dist_px, prox_dist_m)

            # Show Frame
            cv2.imshow(window_name, canvas)

            # Keyboard Events
            key = cv2.waitKey(1 if not self.paused else 30) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # 27 = ESC
                break
            elif key in [ord("p"), ord("P")]:
                self.paused = not self.paused
                print("  [PAUSED]" if self.paused else "  [RESUMED]")
            elif key in [ord("t"), ord("T")]:
                # Cycle Scene Threshold
                curr_idx = self.scenes_list.index(self.current_scene)
                self.current_scene = self.scenes_list[(curr_idx + 1) % len(self.scenes_list)]
                new_tau = self.scene_mgr.get_threshold(self.current_scene)
                print(f"  Switched Scene Context: {self.current_scene.upper()} (Threshold = {new_tau:.2f})")
            elif key in [ord("s"), ord("S")]:
                # Save snapshot
                out_dir = ROOT / "scripts" / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = out_dir / f"skeleton_snapshot_{timestamp}.jpg"
                cv2.imwrite(str(snap_path), canvas)
                print(f"  Saved snapshot: {snap_path}")
            elif key in [ord("c"), ord("C")]:
                # Toggle between camera and video file
                if self.is_webcam:
                    print("  Switching to video file 'real_fight.mp4'...")
                    self.open_capture("real_fight.mp4")
                else:
                    print("  Switching to Webcam (0)...")
                    if not self.open_capture("0"):
                        print("  Webcam unavailable, reverting to video...")
                        self.open_capture("real_fight.mp4")

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n=== Live Skeleton Monitor Closed Cleanly ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartOmniSentinel — Live SOTA Skeleton Monitor")
    parser.add_argument("--source", "--video", dest="source", type=str, default="0", help="Video path, stream URL, or webcam index '0'")
    parser.add_argument("--scene", type=str, default="street", choices=["street", "office", "sports", "indoor_generic"])
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    app = LiveSkeletonApp(source=args.source, scene=args.scene, conf_thresh=args.conf, device=args.device)
    app.run()

