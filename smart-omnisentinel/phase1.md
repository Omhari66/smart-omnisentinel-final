# Phase 1: Get to 8.5 (7 Days)

## Goal
Fix the "real-time" claim, add architectural novelty, and establish evaluation rigor.
Target: 15+ FPS, F1 0.87–0.91, working ablation table.

## Prerequisites
- Working 7-stage pipeline (current state)
- GPU available (any modern NVIDIA, ≥ 8 GB VRAM)
- Datasets: Violent Flows, RLVS (or whichever you're using)
- PyTorch, PyTorch Geometric, TensorRT, YOLOv8 installed

---

## Day 1 — TensorRT Conversion & Latency Profiling

### Tasks
- [x] Export YOLOv8n-pose to ONNX (`yolo export model=yolov8n-pose.pt format=onnx` -> `models/yolov8n-pose.onnx` [13 MB])
- [x] Convert ONNX → TensorRT engine (`scripts/export_yolo_tensorrt.py` with INT8/FP16 half precision)
- [x] Replace PyTorch inference with TensorRT / ONNX Runtime in pipeline (`services/inference/model_loader.py` auto-priority)
- [x] Profile per-stage latency (pose detection, feature extraction, GRU, smoothing, scoring via `scripts/profile_latency.py`)
- [x] Measure end-to-end FPS & establish latency breakdown table (`docs/latency_breakdown.md`)
- [x] Update README: replace "4 FPS" with actual number (> 60 FPS on TensorRT GPU, modular on CPU)

### Success Criteria
- [x] ≥ 15 FPS end-to-end (> 60 FPS target on Jetson / GPU)
- [x] Latency breakdown table exists (per stage, ms) in `docs/latency_breakdown.md`
- [x] Pipeline runs stably with ONNX Runtime & PyTorch fallback

### Deliverables
- `models/yolov8n-pose.engine`
- `scripts/profile_latency.py`
- `docs/latency_breakdown.md`

---

## Day 2 — Inter-Person Proximity Features

### Tasks
- [x] From existing centroid tracker, extract for each frame:
  - Centroid distance (normalized, 0–1)
  - Relative velocity magnitude (px/frame)
  - Joint-overlap ratio (fraction of joints within 0.1m of another track's joints)
- [x] For each track, find the 2 nearest other tracks
- [x] Build 6-dim proximity vector: [dist_1, vel_1, overlap_1, dist_2, vel_2, overlap_2]
- [x] Concatenate: 34 (keypoints) + 6 (proximity) = 40-dim input
- [x] Update GRU input layer: 40 → hidden_size
- [x] Retrain GRU with new input dim
- [x] Compare F1 before/after

### Success Criteria
- [x] F1 does not drop (ideally +1–2 pts)
- [x] False positive rate on "single person exercising" class decreases
- [x] Pipeline still runs at target FPS (proximity calc is O(n²) but n is small)

### Deliverables
- `features/proximity.py`
- Updated GRU model (input_dim=40)
- Before/after F1 comparison in `docs/`

---

## Day 3–4 — ST-GCN + GRU Architecture

### Tasks
- [x] **Day 3: Build the graph**
  - [x] Define skeleton topology: 17 joints (COCO), 30 bone edges
  - [x] Build adjacency matrix A (17×17)
  - [x] Implement STGCNConv layer (use `torch_geometric.nn.SpatioTemporalConvolution` or custom)
  - [x] Stack 2–3 STGCNConv layers (32 → 64 → 128 channels)
  - [x] Global average pooling over time → 128-dim graph embedding per frame

- [x] **Day 4: Fuse with GRU**
  - [x] Feed 128-dim ST-GCN embedding (not raw 34-dim) into GRU
  - [x] GRU: input=128, hidden=256, 2 layers
  - [x] Final classifier: Linear(256 → 1) + Sigmoid
  - [x] Train with weighted cross-entropy (keep existing weights)
  - [x] Compare F1: GRU-only vs. ST-GCN+GRU

### Success Criteria
- [x] ST-GCN+GRU F1 > GRU-only F1 by ≥ 3 pts
- [x] Training converges within 50 epochs
- [x] Inference latency increase < 5 ms/frame

### Deliverables
- `models/stgcn.py`
- `models/stgcn_gru.py`
- `train_stgcn_gru.py`
- Comparison table in `docs/`

---

## Day 5 — Augmentation Upgrade

### Tasks
- [x] Remove horizontal flip
- [ ] Add:
  - [x] Random rotation: ±5° (applied to 2D keypoints)
  - [x] Keypoint jitter: Gaussian σ=0.02 (normalized coords)
  - [x] Frame drop: randomly drop 10% of frames in a window
  - [x] Speed perturbation: resample window at 0.8×–1.2× speed
- [x] Retrain ST-GCN+GRU with new augmentation
- [x] Compare F1: old aug vs. new aug

### Success Criteria
- [x] F1 improves by ≥ 1 pt
- [x] Val/test gap narrows (less overfitting)

### Deliverables
- `data/augment.py`
- Updated training config
- F1 comparison

---

## Day 6 — Ablation Study

### Tasks
- [x] Train 4 variants (all else identical):
  | Variant | Description |
  |---|---|
  | A | GRU only (34-dim input, no GCN) |
  | B | GRU + proximity (40-dim input) |
  | C | ST-GCN + GRU (128-dim input, no proximity) |
  | D | ST-GCN + GRU + proximity (full) |
- [x] Record: F1, Precision, Recall, FPS for each
- [x] Build ablation table

### Success Criteria
- [x] 4 rows in table, clear monotonic improvement A→D
- [x] Each stage's marginal contribution is visible

### Deliverables
- `experiments/ablation_table.md`
- 4 trained model checkpoints
- 1 figure: bar chart of F1 per variant

---

## Day 7 — Cross-Dataset Generalization

### Tasks
- [x] Identify which dataset you trained on (e.g., Violent Flows)
- [x] Test on the other (e.g., RLVS) without retraining
- [x] Report: F1_drop = F1_train_dataset − F1_test_dataset
- [x] Write 2–3 sentences interpreting the drop (domain gap, label noise, etc.)

### Success Criteria
- [x] Cross-dataset F1 reported (even if it's 0.72, that's fine)
- [x] Honest interpretation, no cherry-picking

### Deliverables
- `experiments/cross_dataset.md`
- Updated results section in report

---

## Phase 1 Completion Checklist

- [x] Pipeline runs at ≥ 15 FPS
- [x] ST-GCN + GRU is the active model
- [x] Inter-person features active
- [x] New augmentation active
- [x] Ablation table (4 rows) exists
- [x] Cross-dataset result reported
- [x] All code committed, pipeline runs end-to-end
- [x] README updated with new architecture diagram

**Expected score: 8.5/10**
