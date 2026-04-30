# SmartOmniSentinel — System Architecture Reference

## 1. Design Philosophy

SmartOmniSentinel is not a classifier. It is a **multi-stage probabilistic decision system** where a raw video frame is the least trustworthy signal available. By the time an alert fires, the event has passed through 7 independent validation stages. Every layer exists to reduce the probability that a human responder wastes time on a false alarm.

The three non-negotiable architectural constraints are:

1. **Edge-first inference**: All ML runs on-premises. Raw video never leaves the deployment site. Only event clips (not continuous streams) are transmitted anywhere.
2. **Human-in-the-loop for all borderline cases**: The system never autonomously takes irreversible action. HIGH alerts lock evidence and notify; they do not dispatch police.
3. **Alert pipeline never blocks**: Notifications, DB writes, and evidence extraction are all fire-and-forget asyncio tasks. A slow SMTP server cannot delay the next frame from being processed.

---

## 2. System Layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 0 — INPUT                                            │
│  IP cameras / RTSP streams / webcam / recorded video files  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 1 — INGESTION SERVICE                                │
│  stream_reader.py    Connects to RTSP, samples at target FPS│
│  frame_sampler.py    Adaptive FPS control                   │
│  frame_buffer.py     Circular pre-event buffer (45s)        │
│  health_heartbeat.py Camera liveness monitoring             │
│  reconnect_handler.py Exponential backoff on failure        │
└───────────────────────────┬─────────────────────────────────┘
                            │ Raw frames (numpy arrays)
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 2 — INFERENCE SERVICE                                │
│  detector.py         YOLOv8n — person bounding boxes        │
│  tracker.py          ByteTrack — persistent track IDs       │
│  pose_estimator.py   YOLOv8n-pose — 17-keypoint skeleton    │
│  fall_detector.py    Geometric + binary classifier          │
│  temporal_classifier.py  CNN+GRU 16-frame sliding window    │
│  crowd_analyzer.py   Optical flow + density heuristic       │
│  inference_runner.py Full pipeline orchestrator per camera  │
└───────────────────────────┬─────────────────────────────────┘
                            │ EventCandidate messages
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 3 — EVENT ANALYSIS SERVICE                           │
│  event_analyzer.py   EMA smoother + persistence checker     │
│  event_merger.py     Deduplication — one window per event   │
│  track_history.py    Track consistency FP reduction         │
│  cooldown_manager.py Post-alert suppression state           │
└───────────────────────────┬─────────────────────────────────┘
                            │ Confirmed events
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 4 — RISK ENGINE                                      │
│  signal_extractor.py DB queries for zone + recent alerts    │
│  risk_scorer.py      Weighted multi-signal formula (0–100)  │
│  severity_classifier.py  LOW / MEDIUM / HIGH tier mapping   │
└───────────────────────────┬─────────────────────────────────┘
                            │ Scored incidents
┌────────────────────┬──────▼──────────────────────────────────┐
│  LAYER 5a          │  LAYER 5b                               │
│  ALERT MANAGER     │  EVIDENCE MANAGER                       │
│  alert_dispatcher  │  clip_extractor.py                      │
│  low_handler       │  buffer_flusher.py                      │
│  medium_handler    │  integrity_hasher.py                    │
│  high_handler      │  metadata_writer.py                     │
│  escalation_timer  │  retention_manager.py                   │
│  audit_writer      │                                         │
│  notification_client│                                        │
└────────────────────┴──────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 6 — API GATEWAY (FastAPI)                            │
│  REST endpoints      JWT auth, RBAC, pagination             │
│  WebSocket           Live push to dashboard                 │
│  Signed URLs         Protected evidence clip access         │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 7 — DASHBOARD (React)                                │
│  Live camera grid    Incident review queue                  │
│  Alert feed          Evidence player                        │
│  Analytics           System health                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Event Bus Architecture

The internal event bus connects services without direct function calls.

```
CHANNEL: frames.{camera_id}
  Publisher:  ingestion/stream_reader.py
  Consumer:   inference/inference_runner.py

CHANNEL: event_candidates.{camera_id}
  Publisher:  inference/inference_runner.py
  Consumer:   event_analysis/event_analyzer.py

CHANNEL: confirmed_events
  Publisher:  event_analysis/event_analyzer.py
  Consumer:   risk_engine/severity_classifier.py

CHANNEL: scored_incidents
  Publisher:  risk_engine/severity_classifier.py
  Consumers:  alert_manager/alert_dispatcher.py
              evidence_manager/clip_extractor.py (triggers clip start)

CHANNEL: alert_actions
  Publisher:  alert_manager/alert_dispatcher.py
  Consumers:  notification/ (email, webhook, sms)
              api/websocket_manager.py (push to dashboard)

CHANNEL: camera_health
  Publisher:  ingestion/health_heartbeat.py
  Consumer:   health_monitor/camera_health.py

CHANNEL: system_health
  Publisher:  health_monitor/service_health.py
  Consumer:   health_monitor/health_reporter.py → WebSocket
```

**MVP mode**: `asyncio.Queue` (zero infrastructure, single-process)  
**V2 mode**: Redis Streams (multi-process, multi-machine, no code changes in services)

---

## 4. ML Pipeline Detail

### 4.1 Input and Feature Extraction

Each video frame produces a **55-dimensional feature vector** per tracked person:

```
[0:34]   17 keypoint (x, y) coordinates, normalized to [0,1]
[34:51]  17 keypoint confidence scores
[51]     Track motion magnitude (normalized)
[52]     Scene optical flow magnitude (normalized)
[53]     Min inter-person distance (normalized)
[54]     Avg inter-person distance (normalized)
```

These vectors form a **(16, 55) sliding window** — 16 frames at 4fps effective = 4 seconds of behavioral context.

### 4.2 Model Architecture

```
Input: (batch, 16, 55)
  ↓
Linear(55 → 128) + ReLU + Dropout(0.3)   [per-frame encoding]
  ↓
GRU(128 → 256, 2 layers, dropout=0.3)    [temporal modeling]
  ↓
Last hidden state: (batch, 256)
  ↓
Linear(256 → 128) + ReLU + Dropout(0.3)
  ↓
Linear(128 → 3)                          [logits]
  ↓
Softmax → [normal, violent_interaction, person_distress]
```

**Why GRU over LSTM**: GRU has 25% fewer parameters with equivalent performance on sequences under 32 frames. Critical for Jetson memory budget.

**Why not Video Transformer**: TimeSformer requires 8–32 frame batches processed as a unit. Inference latency on Jetson AGX at 720p = 180–400ms per clip, which exceeds our 250ms target. GRU processes frames incrementally with 5–8ms per step.

### 4.3 Confidence Smoothing

Raw softmax output is noisy. A single occluded frame can drop confidence from 0.87 to 0.31.

Exponential Moving Average:
```
smoothed_t = α × raw_t + (1 − α) × smoothed_{t−1}
α = 0.35 (default)
```

**Persistence requirement**: smoothed confidence must exceed threshold for **12 consecutive frames** (~3 seconds at 4fps) before event is confirmed. This single rule eliminates ~60–70% of false positives.

### 4.4 Fall Detection (Hybrid)

Falls use a two-stage approach because fall training data is scarce and often staged:

**Stage 1 — Geometric trigger** (high recall):
- Body bounding box aspect ratio crosses 1.4 → 1.0 (standing → fallen)
- Head keypoint Y drops >25% of body height within 8 frames
- Person must have been standing for ≥10 frames first

**Stage 2 — Binary classifier confirmation** (precision):
- Input: flattened keypoints + confidences (51 features)
- Output: fall probability
- Only runs when geometric trigger fires
- If classifier unavailable: geometric trigger alone fires with confidence 0.72

---

## 5. Risk Scoring Formula

```python
RISK = clamp(
    30 × confidence_factor          # Smoothed classifier confidence
  + 15 × duration_factor           # min(duration / 30s, 1.0)
  + 10 × people_count_factor       # min(people / 5, 1.0)
  + 10 × motion_factor             # min(flow_magnitude / 8.0, 1.0)
  + 10 × recent_alert_factor       # 1.0 if alert in last 10 min, else 0
  + 15 × zone_sensitivity          # Configured per zone (0.0–1.0)
  +  5 × time_of_day_factor        # 1.0 if 10pm–6am, else 0.3
  +  5 × human_confirmation_bonus  # 1.0 if reviewer confirmed, else 0
  , 0, 100
)
```

**Severity tiers** (configurable per deployment):
- Score 0–29: LOW → review queue, no alert
- Score 30–64: MEDIUM → security notification, escalation timer
- Score 65–100: HIGH → emergency workflow, evidence lock

---

## 6. Three-Tier Alert Logic

### LOW (score < 30)
- Incident record created, status: `PENDING_REVIEW`
- 30s clip written to staging storage
- Entry added to reviewer queue on dashboard
- Auto-deleted after 72h if not confirmed
- **No external notification**

### MEDIUM (score 30–64)
- Incident created, status: `ACTIVE_REVIEW`
- Reviewer notified via webhook + email
- Dashboard shows amber warning
- 60s clip locked in evidence staging (not auto-deleted)
- **Escalation timer starts** (default 5 minutes)
- If not reviewed within timer: auto-escalates to HIGH

### HIGH (score ≥ 65)
- Incident created, status: `EMERGENCY`
- Evidence clip immediately locked (immutable)
- Full notification: webhook + email + SMS (if configured)
- Dashboard full-screen red alert
- **Cooldown set** on event analyzer (default 5 minutes)
- Event merger window closed

**What never happens automatically**: police/authority dispatch. The dashboard provides an "ESCALATE TO AUTHORITIES" button with a confirmation step and audit trail.

---

## 7. Evidence Chain of Custody

```
t=0    Event confirmed by event analyzer
t=0    Pre-event buffer (last 45s) flushed to staging/INC-xxx_pre.mp4
t=0+15s Post-event capture ends, staging/INC-xxx_post.mp4 written
t=15s  Pre + post merged → clips/INC-xxx_full.mp4
t=15s  SHA-256 hash computed and stored in DB + manifest
t=15s  JSON manifest written to manifests/INC-xxx.json
t=15s  EvidenceClip DB record created with hash, path, retention tag
t=15s  EVIDENCE_READY WebSocket event pushed to dashboard
       All HIGH clips: is_locked=True (cannot be auto-deleted)
       All clip access logged to append-only audit_log table
       Clips served only via 15-minute signed URLs (not public paths)
```

---

## 8. False Positive Reduction Stack

Seven layers applied in order:

| Layer | Mechanism | FP Reduction (est.) |
|-------|-----------|---------------------|
| 1 | EMA confidence smoothing (α=0.35) | 20–30% |
| 2 | Persistence threshold (12 frames) | 40–50% |
| 3 | Multi-signal risk score (not just confidence) | 15–20% |
| 4 | Zone-specific thresholds (gym=0.85, corridor=0.65) | 10–15% |
| 5 | Track history consistency multiplier | 5–10% |
| 6 | Event deduplication (one window per camera+type) | 30–40% |
| 7 | Cooldown after HIGH alert (5 min) | Alert storm prevention |

Confusing cases handled explicitly:
- **Dancing/sports**: Zone suppression + bilateral symmetry heuristic
- **Playful pushing**: Fails persistence threshold (<3s)
- **Hugging**: No striking motion vectors; pose shows face proximity
- **Low light**: Detector confidence threshold raised when SSIM < 0.3
- **Occlusion**: Classifier abstains if >40% keypoints invisible
- **Camera shake**: Global motion detected → suppress crowd aggression

---

## 9. Deployment Tiers

| Tier | Hardware | FPS | Cameras | Notes |
|------|----------|-----|---------|-------|
| Demo | Laptop (CPU) | 6–8 | 1 | YOLOv8n on CPU, no GPU needed |
| Workstation | NVIDIA RTX | 20–25 | 1–4 | CUDA path, Docker Compose |
| Edge | Jetson Orin NX | 12–15 | 1–2 | TensorRT FP16, 10W budget |
| NVR Integration | Edge box + NVR | 8–12 | 4–8 | RTSP from NVR, edge stores clips |

---

## 10. Security Model

| Concern | Mechanism |
|---------|-----------|
| API authentication | JWT (15-min access token, 7-day refresh) |
| Camera node auth | API key (write-only scope) |
| Evidence access | HMAC signed URLs (15-min expiry) |
| Role isolation | ADMIN / SUPERVISOR / GUARD / VIEWER |
| Audit trail | Append-only DB table, no UPDATE/DELETE |
| Evidence integrity | SHA-256 hash at creation, verified on access |
| Raw video privacy | Inference on-premises, never transmitted |
| Face privacy | Optional blur on export (V2 feature) |
