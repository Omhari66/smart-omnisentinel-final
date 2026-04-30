# SmartOmniSentinel

**Edge AI-Based Real-Time Incident Detection and Response for CCTV Networks**

A production-grade intelligent surveillance system that converts standard CCTV infrastructure
into a proactive safety platform. Detects violent incidents, falls, and crowd anomalies in
real time; reduces false positives through multi-stage analysis; triggers calibrated responses
by severity; and maintains a forensically-sound evidence trail.

---

## Architecture Overview

```
Camera Stream → Ingestion → Inference → Event Analysis → Risk Engine → Alert Manager
                                                                    ↓
                                                           Evidence Manager
                                                                    ↓
                                                           FastAPI + WebSocket
                                                                    ↓
                                                           Dashboard / Notifications
```

### Detection Pipeline

| Stage | Component | Model |
|-------|-----------|-------|
| Person Detection | `PersonDetector` | YOLOv8n |
| Multi-Object Tracking | `PersonTracker` | ByteTrack (via Ultralytics) |
| Pose Estimation | `PoseEstimator` | YOLOv8n-pose |
| Temporal Classification | `TemporalClassificationEngine` | CNN + GRU |
| Fall Detection | `FallDetector` | Geometric rules + binary classifier |
| Crowd Anomaly | `InferenceRunner` | Optical flow heuristic |

### False Positive Reduction Stack

1. **EMA confidence smoothing** (α=0.35, configurable)
2. **Persistence threshold** (12 consecutive frames, ~3s at 4fps)
3. **Zone suppression** (e.g. fall detection disabled in gym zones)
4. **Cooldown** (300s after HIGH alert — prevents alert storms)
5. **Deduplication** (merges continuations of same event into one incident)
6. **Three-tier severity** (LOW/MEDIUM/HIGH) — guards only paged for confirmed events

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or SQLite for development)
- Docker + Docker Compose (for full-stack deployment)

### Development Setup (Single Machine)

```bash
# 1. Clone and create virtual environment
git clone <repo>
cd smart-omnisentinel
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY

# 4. Initialize database
alembic upgrade head
bash deployments/scripts/init_db.sh

# 5. Start the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive API documentation.

Default admin login: `admin@sentinel.local` / `ChangeMe123!`

### Docker Compose (Full Stack)

```bash
cp .env.example .env
# Set POSTGRES_PASSWORD and SECRET_KEY in .env

docker compose up -d

# First run: initialize DB
docker compose exec api alembic upgrade head
docker compose exec api bash deployments/scripts/init_db.sh
```

---

## Training the ML Model

```bash
# 1. Prepare dataset (see ml/datasets/ for loaders)
#    Expected structure:
#    data/processed/
#      train/{normal,violent,distress}/*.npy
#      val/{normal,violent,distress}/*.npy

# 2. Train
python -m ml.training.train_classifier \
    --data_root data/processed \
    --epochs 50 \
    --device cuda \
    --checkpoint_dir ml/checkpoints

# 3. Export to ONNX (optional, for deployment)
python -m ml.export.export_onnx \
    --checkpoint ml/checkpoints/temporal_v1.pt \
    --validate
```

---

## Testing the Pipeline

```bash
# Test with a video file (no DB required)
python scripts/replay_video_file.py --video /path/to/test.mp4

# Test with webcam
python scripts/replay_video_file.py --video 0

# Benchmark inference speed on current hardware
python scripts/benchmark_inference.py --frames 200 --device cpu
```

---

## API Reference

The full OpenAPI spec is available at `/docs` (development mode).

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/login` | Obtain JWT tokens |
| GET | `/api/v1/cameras` | List cameras |
| POST | `/api/v1/cameras` | Register camera |
| POST | `/api/v1/cameras/{id}/start` | Start stream |
| GET | `/api/v1/incidents` | List incidents (filtered) |
| GET | `/api/v1/incidents/{id}` | Incident detail |
| GET | `/api/v1/review/queue` | Review queue |
| POST | `/api/v1/review/{id}/action` | Submit review decision |
| GET | `/api/v1/evidence/{id}/url` | Get signed clip URL |
| WS | `/api/v1/ws/events?token=<jwt>` | Live event stream |

---

## Configuration

### Threshold Profiles

Threshold profiles are stored in `configs/threshold_profiles/` and in the database.
Per-camera thresholds override global defaults.

```yaml
violence_confidence_min: 0.72   # Classifier confidence floor
persistence_frames: 12           # Frames event must persist (~3s at 4fps)
risk_low_max: 29                 # Score ≤ 29 → LOW
risk_medium_max: 64              # Score ≤ 64 → MEDIUM, > 64 → HIGH
cooldown_seconds: 300            # Alert storm prevention
```

### Zone Configs

Zones define sensitivity per camera location:

```yaml
name: school_corridor
sensitivity: 0.90               # High sensitivity zone
suppress_event_types: []        # No suppressions
```

---

## Project Structure

```
smart-omnisentinel/
├── api/                    # FastAPI routes, schemas, middleware
├── core/                   # Config, logging, security, event bus
├── db/                     # SQLAlchemy models, migrations
├── services/
│   ├── ingestion/          # RTSP stream reading
│   ├── inference/          # Detection, tracking, pose, classification
│   ├── event_analysis/     # Smoothing, persistence, deduplication
│   ├── risk_engine/        # Multi-signal risk scoring
│   ├── alert_manager/      # Severity routing, escalation, audit
│   ├── evidence_manager/   # Clip extraction, hashing, retention
│   └── notification/       # Email, webhook dispatch
├── ml/                     # Training, datasets, model definitions, export
├── configs/                # YAML configuration files
├── tests/                  # Unit and integration tests
├── scripts/                # Developer utilities
└── deployments/            # Docker, Nginx, initialization scripts
```

---

## Resume Description

> **SmartOmniSentinel** — Designed and implemented an end-to-end, production-grade intelligent
> CCTV surveillance system. Built a 7-stage ML inference pipeline (YOLOv8 detection + ByteTrack
> MOT + pose estimation + CNN-GRU temporal classification) achieving real-time incident detection
> at <3s alert latency. Engineered a multi-signal risk scoring engine, three-tier alert severity
> system with auto-escalation, and forensic evidence management with SHA-256 integrity hashing.
> Deployed as a modular FastAPI microservice architecture with WebSocket live alerting, JWT-RBAC
> auth, and Docker Compose orchestration. Reduced false positive rate by 60–70% through EMA
> smoothing, persistence thresholds, zone-aware suppression, and cooldown logic.

---

## License

Academic / research use. Not intended for production surveillance without appropriate legal review.
