# SmartOmniSentinel — Deployment Guide

---

## 1. Deployment Tiers

| Tier | Target | Hardware | GPU | Cameras | Use case |
|------|--------|----------|-----|---------|---------|
| Demo | Laptop | Any modern laptop | Optional | 1 (webcam) | Faculty demo, investor pitch |
| Workstation | Local server | i7/i9 + RTX 30/40 | Required for >2 cameras | 1–4 | Lab, small office |
| Edge Box | Jetson Orin NX | Jetson Orin NX 16GB | Yes (integrated) | 1–4 | Production edge deployment |
| NVR Integration | Jetson + NVR | Jetson + existing NVR | Yes | 4–16 | Enterprise / campus |

---

## 2. Demo Mode (Single Process, No Docker)

Fastest to get running. All services run in one process.

```bash
# Prerequisites: Python 3.11, pip install -r requirements.txt

# Set environment
export APP_ENV=development
export SECRET_KEY=demo-key-change-me
export DB__URL=sqlite+aiosqlite:///demo.db    # SQLite, no postgres needed
export REDIS__ENABLED=false

# Run migrations on SQLite
alembic upgrade head

# Start
uvicorn api.main:app --reload

# In another terminal, test with webcam:
python scripts/replay_video_file.py --video 0 --camera_id demo-cam
```

Demo mode uses SQLite (no PostgreSQL needed) and runs all workers as asyncio tasks within the FastAPI process. No Docker required.

---

## 3. Docker Compose (Workstation / Server)

### 3.1 Prepare environment

```bash
cp .env.example .env

# Edit .env — required changes:
# SECRET_KEY=<python -c "import secrets; print(secrets.token_hex(32))">
# SIGNED_URL__SECRET=<another random 32-char hex>
# POSTGRES_PASSWORD=choose-a-strong-password
```

### 3.2 Build and start

```bash
docker compose up -d --build

# Watch startup logs
docker compose logs -f api

# Expected: "sentinel_ready" log line within 30 seconds
```

### 3.3 Initialize database

```bash
# Migrations run automatically on API startup via CMD in docker-compose.yml
# To seed default data manually:
docker compose exec api bash deployments/scripts/init_db.sh
```

### 3.4 Verify

```bash
curl http://localhost/api/v1/health/ping
# {"status":"ok","ts":"..."}

curl http://localhost/docs
# Swagger UI
```

### 3.5 Configure for GPU

Edit `.env`:
```bash
INFERENCE__DEVICE=cuda
```

Edit `docker-compose.yml` — uncomment the deploy/resources/reservations block:
```yaml
inference_worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Requires NVIDIA Container Runtime: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

---

## 4. Jetson Orin NX (Edge Deployment)

### 4.1 Jetson setup

```bash
# Install JetPack 6.x (includes CUDA, cuDNN, TensorRT)
# Verify TensorRT:
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Install Python dependencies (Jetson-specific torch build)
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 \
    torch torchvision

pip3 install -r requirements.txt
```

### 4.2 Build TensorRT engine (on Jetson)

```bash
# First export ONNX on any machine, then copy to Jetson:
scp ml/checkpoints/temporal_v1.onnx jetson:/home/sentinel/smart-omnisentinel/ml/checkpoints/

# On Jetson — build TensorRT FP16 engine:
python3 -m ml.export.export_tensorrt \
    --onnx ml/checkpoints/temporal_v1.onnx \
    --output ml/checkpoints/temporal_v1.trt \
    --fp16

# Alternative using trtexec (faster):
trtexec --onnx=ml/checkpoints/temporal_v1.onnx \
        --saveEngine=ml/checkpoints/temporal_v1.trt \
        --fp16
```

### 4.3 Deploy with edge Docker Compose

```bash
# On Jetson:
docker compose -f docker-compose.yml -f docker-compose.edge.yml up -d

# Edge compose: no dashboard, TensorRT enabled, reduced FPS, memory limits
```

### 4.4 Performance tuning for Jetson

Set in `.env` on Jetson:
```bash
INFERENCE__DEVICE=cuda
INFERENCE__USE_TENSORRT=true
INFERENCE__EFFECTIVE_FPS=3          # 3fps instead of 4 for thermal headroom
INFERENCE__DETECTION_SKIP_FRAMES=5  # Detect every 5 frames
STORAGE__MAX_DISK_GB=20.0           # Conservative storage limit
```

Expected performance (Jetson Orin NX 16GB):
- YOLOv8n (TensorRT FP16, 480p): ~8ms
- Pose estimation: ~12ms
- GRU temporal model (ONNX): ~4ms
- Total per-frame: ~25–30ms → ~33 FPS throughput (well above 3fps target)

### 4.5 Thermal management

```bash
# Set Jetson to max performance mode:
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperature during load:
watch -n 1 "tegrastats | grep -o 'CPU@[0-9.]*C\|GPU@[0-9.]*C'"
```

---

## 5. NVR Integration Mode

For sites with existing NVR (Network Video Recorder):

### 5.1 Architecture

```
[IP Cameras] → [NVR (Synology/Hikvision/etc.)] → [RTSP output]
                                                         ↓
                                              [Jetson Edge Box]
                                              [SmartOmniSentinel]
                                                         ↓
                                              [Evidence storage]
                                              [Alert system]
```

The NVR handles raw video storage. SmartOmniSentinel connects to the NVR's RTSP output for live streams.

### 5.2 Finding RTSP URLs

Common NVR RTSP formats:
```
Synology:     rtsp://admin:pass@192.168.1.10:554/videoMain
Hikvision:    rtsp://admin:pass@192.168.1.10:554/Streaming/Channels/1
Dahua:        rtsp://admin:pass@192.168.1.10:554/cam/realmonitor?channel=1&subtype=0
Generic ONVIF: rtsp://user:pass@ip:554/onvif/live
```

Test connectivity first:
```bash
python scripts/test_rtsp_stream.py "rtsp://admin:pass@192.168.1.10:554/videoMain"
```

---

## 6. HTTPS / TLS Configuration

For production, add TLS termination at nginx:

```nginx
# In deployments/nginx/nginx.conf, add server block:
server {
    listen 443 ssl;
    server_name sentinel.yoursite.com;
    
    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # ... same location blocks as HTTP config
}
```

Use Let's Encrypt (certbot) or your organization's certificates.

---

## 7. Production Checklist

Before going live, verify:

**Security:**
- [ ] `SECRET_KEY` is a random 32+ char hex string (not the default)
- [ ] `SIGNED_URL__SECRET` is different from `SECRET_KEY`
- [ ] `POSTGRES_PASSWORD` is strong and not the default
- [ ] `.env` file is not committed to git (check `.gitignore`)
- [ ] Admin password changed from default (`ChangeMe123!`)
- [ ] HTTPS configured (not HTTP) if accessible over the internet
- [ ] Swagger UI disabled in production: `APP_ENV=production`

**Storage:**
- [ ] Evidence storage mounted to persistent volume (not container filesystem)
- [ ] Available disk space ≥ `STORAGE__MAX_DISK_GB`
- [ ] Retention policies reviewed and appropriate for site regulations

**Operations:**
- [ ] Alert email recipients configured and tested
- [ ] Webhook URLs configured and tested
- [ ] All cameras tested with `test_rtsp_stream.py`
- [ ] Threshold profiles tuned for each zone
- [ ] At least one SUPERVISOR and one GUARD user created
- [ ] Health monitoring alert thresholds verified

**ML:**
- [ ] Model trained or fine-tuned on site-specific data
- [ ] False positive rate validated on 1-week of real footage before live alerting
- [ ] Fall detection geometric rules validated (test with actual fall scenarios)

---

## 8. Monitoring and Operations

### 8.1 Log monitoring

```bash
# Docker:
docker compose logs -f api inference_worker alert_worker

# Structured log search (if using log aggregator):
# Find all HIGH alerts in last hour:
# level=warning action=alert_high_emergency | last 1h
```

### 8.2 Health check

```bash
# External monitoring (Uptime Robot / Nagios / Prometheus):
GET /api/v1/health/ping

# Detailed health (authenticated):
GET /api/v1/health
```

### 8.3 Database maintenance

```bash
# Check incident table size:
docker compose exec postgres psql -U sentinel -c \
    "SELECT count(*), pg_size_pretty(pg_total_relation_size('incidents')) FROM incidents;"

# Manual evidence cleanup (if retention_manager not running):
docker compose exec api python -c "
import asyncio
from services.evidence_manager.retention_manager import RetentionManager
asyncio.run(RetentionManager()._purge_expired())
"
```

### 8.4 Model updates

```bash
# After retraining:
# 1. Copy new checkpoint to ml/checkpoints/temporal_v2.pt
# 2. Update .env: INFERENCE__CLASSIFIER_CHECKPOINT=ml/checkpoints/temporal_v2.pt
# 3. Restart inference worker:
docker compose restart inference_worker
```
