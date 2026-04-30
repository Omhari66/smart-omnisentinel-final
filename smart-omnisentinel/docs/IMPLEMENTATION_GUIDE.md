# SmartOmniSentinel — Implementation Guide

Step-by-step guide from zero to a running system. Follow phases in order.
Each phase ends with a testable milestone.

---

## Prerequisites

```bash
# Required
python 3.11+
postgresql 16+
docker + docker compose
git

# For ML training
nvidia-smi   # if using GPU (optional for MVP)

# For Jetson deployment
jetpack 6.x  # includes TensorRT, CUDA
```

---

## Phase A — Foundation (Day 1)

### A1. Clone and set up environment

```bash
git clone <repo> smart-omnisentinel
cd smart-omnisentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### A2. Configure environment

```bash
cp .env.example .env
# Edit .env minimum required fields:
#   SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
#   SIGNED_URL__SECRET=<another random string>
#   DB__URL=postgresql+asyncpg://sentinel:yourpass@localhost:5432/sentinel
```

### A3. Start database

```bash
docker compose up -d postgres
# Wait 10 seconds for postgres to be ready
docker compose exec postgres pg_isready -U sentinel
```

### A4. Initialize database

```bash
bash deployments/scripts/init_db.sh
# This runs Alembic migrations + seeds default data
```

### A5. Start API server

```bash
uvicorn api.main:app --reload
# Open http://localhost:8000/docs
```

**Milestone A**: Swagger UI loads, `/api/v1/health/ping` returns `{"status":"ok"}`.

---

## Phase B — Inference Pipeline (Day 2–3)

### B1. Test YOLOv8n availability

```bash
python3 -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print('OK')"
# First run downloads yolov8n.pt (~6MB)
```

### B2. Run pipeline on a test video

```bash
# Download any test video (MP4 or AVI)
python scripts/replay_video_file.py \
    --video /path/to/test.mp4 \
    --camera_id test-cam-001 \
    --fps 4

# Expected output:
# Models loaded.
# Progress: 50/XXX (X%) | Processed X frames in Xs
# [CANDIDATE] frame=52 type=VIOLENT_INTERACTION conf=0.xxx  (if fighting detected)
```

### B3. Benchmark inference speed

```bash
python scripts/benchmark_inference.py --frames 100 --resolution 480p
# Target: >= 4 FPS (system only needs 4fps effective)
```

**Milestone B**: Pipeline runs on test video without errors. Candidates are printed to stdout.

---

## Phase C — Train the Classifier (Day 3–7)

### C1. Download training data

**Option 1 — RWF-2000** (recommended, violence detection):
```bash
# Request access at: https://github.com/mchengny/RWF2000-Video-Database
# Download and extract to: data/raw/RWF-2000/
```

**Option 2 — Custom data** (if you have annotated CCTV footage):
```bash
# Create CSV with columns: video_path,label,start_frame,end_frame,notes
# Labels: normal, violent, distress, fall, crowd_aggression
```

### C2. Extract features

```bash
# For RWF-2000:
python -m ml.datasets.rwf2000_loader \
    --input data/raw/RWF-2000 \
    --output data/processed

# For custom data:
python -m ml.datasets.custom_loader \
    --annotations data/annotations/my_incidents.csv \
    --output data/processed

# Check distribution:
python -m ml.training.dataset_builder --output data/processed
```

### C3. Train

```bash
python -m ml.training.train_classifier \
    --data_root data/processed \
    --epochs 50 \
    --batch_size 32 \
    --device cpu   # or cuda
    --checkpoint_dir ml/checkpoints

# Monitor: watch for "Best checkpoint saved" lines
# Target: val mean_F1 > 0.70 on RWF-2000
```

### C4. Evaluate

```bash
python -m ml.training.evaluate \
    --checkpoint ml/checkpoints/temporal_v1.pt \
    --data_root data/processed/val

# Expected output: per-class precision/recall/F1 + confusion matrix
```

### C5. Export to ONNX

```bash
python -m ml.export.export_onnx \
    --checkpoint ml/checkpoints/temporal_v1.pt \
    --output ml/checkpoints/temporal_v1.onnx \
    --validate

# Should print: "✓ All tests passed"
```

**Milestone C**: Trained model with val mean_F1 > 0.65. ONNX export validated.

---

## Phase D — Alert and Evidence System (Day 5–8)

### D1. Register a test camera via API

```bash
# Login first
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"admin@sentinel.local","password":"ChangeMe123!"}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Register camera
curl -X POST http://localhost:8000/api/v1/cameras \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Test Camera",
        "stream_url": "0",
        "location": "Test Lab",
        "fps_target": 4
    }'
```

### D2. Inject a test incident

```bash
# Get camera ID from step D1, then:
python scripts/generate_test_incident.py \
    --camera_id <camera-uuid> \
    --severity HIGH \
    --event_type VIOLENT_INTERACTION \
    --risk_score 78
```

### D3. Check WebSocket events

```javascript
// In browser console:
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/events?token=YOUR_TOKEN');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

### D4. Verify incident in API

```bash
curl -H "Authorization: Bearer $TOKEN" \
    http://localhost:8000/api/v1/incidents

# Should show the injected test incident
```

**Milestone D**: Test incident created, appears in API response and WebSocket.

---

## Phase E — Full Docker Deployment (Day 8–10)

### E1. Build and start all services

```bash
docker compose up -d --build
# Wait ~60 seconds for all services to be healthy

docker compose ps          # All should be "Up (healthy)"
docker compose logs api    # Check for errors
```

### E2. Run database migrations inside container

```bash
docker compose exec api alembic upgrade head
docker compose exec api bash deployments/scripts/init_db.sh
```

### E3. Test end-to-end via Docker

```bash
# Health check
curl http://localhost:80/api/v1/health/ping

# Login
curl -X POST http://localhost:80/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"admin@sentinel.local","password":"ChangeMe123!"}'
```

**Milestone E**: Full Docker stack running, API accessible through nginx.

---

## Phase F — Live Camera Integration (Day 10+)

### F1. Test RTSP connectivity

```bash
python scripts/test_rtsp_stream.py rtsp://192.168.1.20:554/stream1
# Should print: "✓ Stream is healthy"
```

### F2. Register live camera

```bash
curl -X POST http://localhost:8000/api/v1/cameras \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Front Door",
        "stream_url": "rtsp://192.168.1.20:554/stream1",
        "location": "Building Entrance",
        "fps_target": 4
    }'
```

### F3. Start stream

```bash
CAMERA_ID=<uuid-from-step-F2>

curl -X POST http://localhost:8000/api/v1/cameras/$CAMERA_ID/start \
    -H "Authorization: Bearer $TOKEN"
```

**Milestone F**: Live camera stream processing. Events appear in dashboard.

---

## Common Issues and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: ultralytics` | Not installed | `pip install ultralytics` |
| `Connection refused` on DB | Postgres not ready | Wait 10s, check `docker compose ps` |
| `Checkpoint not found` warning | Model not trained | Run Phase C or set `INFERENCE__CLASSIFIER_CHECKPOINT` to valid path |
| WebSocket immediately closes | Invalid JWT token | Re-login and use fresh access token |
| `0 syntax errors` but import fails | Missing dependency | Check `requirements.txt`, reinstall |
| Evidence clips not created | `evidence_storage/` missing | `mkdir -p evidence_storage/{staging,clips,manifests}` |
| Inference very slow | Running on CPU at 1080p | Set `INFERENCE__EFFECTIVE_FPS=2` or resize frames before inference |

---

## Development Tips

**Running tests:**
```bash
pytest tests/unit/ -v                    # Fast unit tests
pytest tests/integration/ -v             # Needs running postgres
pytest tests/e2e/ -v -s                  # End-to-end (slowest)
pytest --cov=services --cov-report=html  # Coverage report
```

**Hot reloading in development:**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
# Source mounted as volume — code changes reload automatically
```

**Resetting the database (development only):**
```bash
docker compose exec postgres psql -U sentinel -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
alembic upgrade head
bash deployments/scripts/init_db.sh
```

**Viewing logs:**
```bash
docker compose logs -f api            # API logs
docker compose logs -f inference_worker  # Inference logs
```
