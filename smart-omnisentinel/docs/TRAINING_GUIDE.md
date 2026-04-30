# SmartOmniSentinel — Training Guide

---

## 1. Dataset Strategy Overview

The classifier needs data from three behavioral categories:

| Label | Description | Source datasets |
|-------|-------------|-----------------|
| `normal` | All non-incident activity | RWF-2000 NonFight, UCF-Crime Normal, custom |
| `violent` | Fighting, assault, aggression | RWF-2000 Fight, UCF-Crime Fighting/Assault/Robbery |
| `distress` | Fall, collapse, incapacitation | UR Fall Detection, UCF-Crime Abuse |

A separate binary classifier handles `fall` detection (see Section 5).

**Critical warning**: Internet violence datasets contain staged/acted footage that generalizes poorly to real CCTV. Always validate with site-specific data before deploying.

---

## 2. Recommended Dataset Combination

For a production-quality model, combine at minimum:

```
data/processed/
├── train/
│   ├── normal/      Target: ≥ 3000 windows
│   ├── violent/     Target: ≥ 1500 windows
│   └── distress/    Target: ≥ 500 windows
└── val/
    ├── normal/      Target: ≥ 500 windows
    ├── violent/     Target: ≥ 300 windows
    └── distress/    Target: ≥ 100 windows
```

**Minimum viable dataset**: 500 train windows total (lower quality, expect val F1 ~0.60).  
**Good dataset**: 5000+ train windows across all classes (expect val F1 ~0.75+).

---

## 3. Dataset Download and Preparation

### 3.1 RWF-2000 (Primary — violence detection)

1. Request access: https://github.com/mchengny/RWF2000-Video-Database  
2. Extract to `data/raw/RWF-2000/`
3. Process:

```bash
python -m ml.datasets.rwf2000_loader \
    --input data/raw/RWF-2000 \
    --output data/processed
```

Expected output: ~6000 train windows, ~1500 val windows across normal/violent.

### 3.2 UR Fall Detection Dataset (Fall/distress)

1. Download: http://fenix.ur.edu.pl/~mkepski/ds/uf.html  
2. Extract to `data/raw/urfall/`
3. Process:

```bash
python -m ml.datasets.urfall_loader \
    --input data/raw/urfall \
    --output data/processed
```

Note: UR Fall uses depth + RGB. We use RGB frames only.

### 3.3 UCF-Crime (Additional violence + distress)

1. Request access: https://www.crcv.ucf.edu/projects/real-world/  
2. Extract to `data/raw/UCF_Crimes/`
3. Process (relevant classes only):

```bash
python -m ml.datasets.ucfcrime_loader \
    --root data/raw/UCF_Crimes \
    --output data/processed \
    --annotation_file data/raw/UCF_Crimes/Annotations/Temporal_Anomaly_Annotation.txt
```

**Important**: UCF-Crime has label ambiguity. Review extracted windows before training.

### 3.4 Custom CCTV Footage (Domain adaptation — highly recommended)

Custom data from your deployment site is the highest-value addition.
Even 50–100 annotated clips from your own cameras will significantly improve performance.

```bash
# 1. Create annotations CSV:
# video_path,label,start_frame,end_frame,notes
# /recordings/incident_2024_01.mp4,violent,150,320,two people fighting
# /recordings/normal_lobby.mp4,normal,0,1000,routine foot traffic

# 2. Process:
python -m ml.datasets.custom_loader \
    --annotations data/annotations/my_site.csv \
    --output data/processed
```

### 3.5 Build unified dataset

```bash
# Combines all processed sources, prints distribution stats:
python -m ml.training.dataset_builder --output data/processed
```

---

## 4. Training the Temporal Classifier

### 4.1 Quick start (CPU, small dataset)

```bash
python -m ml.training.train_classifier \
    --data_root data/processed \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-3 \
    --device cpu \
    --checkpoint_dir ml/checkpoints
```

Expected time: ~2 hours for 3000 samples on modern CPU.

### 4.2 Full training (GPU recommended)

```bash
python -m ml.training.train_classifier \
    --data_root data/processed \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --device cuda \
    --checkpoint_dir ml/checkpoints
```

Expected time: ~20 minutes on RTX 3060, ~8 minutes on RTX 4090.

### 4.3 Training output

```
Epoch 001/050 | Train loss=1.0842 acc=0.421 | Val loss=0.9743 acc=0.481 mean_f1=0.332 | 18.3s
Epoch 005/050 | Train loss=0.7213 acc=0.698 | Val loss=0.6841 acc=0.718 mean_f1=0.651 | 17.1s
  ↑ Best checkpoint saved (mean F1=0.651)
...
Epoch 050/050 | Train loss=0.2341 acc=0.921 | Val loss=0.4122 acc=0.863 mean_f1=0.847 | 17.0s
  ↑ Best checkpoint saved (mean F1=0.847)

Training complete. Best val mean_F1: 0.847
Checkpoint: ml/checkpoints/temporal_v1.pt
```

### 4.4 Target metrics (with RWF-2000 + custom data)

| Metric | Acceptable | Good | Excellent |
|--------|-----------|------|-----------|
| Val mean F1 | > 0.65 | > 0.75 | > 0.85 |
| Violence precision | > 0.70 | > 0.80 | > 0.90 |
| Violence recall | > 0.65 | > 0.75 | > 0.85 |
| False positive rate | < 0.20 | < 0.12 | < 0.07 |

---

## 5. Fall Detector (Optional Classifier Component)

The fall detector works without a trained classifier (geometric rules alone achieve ~72% confidence). Training the binary classifier improves precision.

```bash
# Requires urfall data already processed into data/processed/
# The fall classifier uses 51-dim single-frame features, not 16-frame windows

# Training script (adapt train_classifier.py):
# - Change model: FallBinaryClassifier() instead of TemporalClassifier()
# - Change classes: ["fall", "normal"] binary
# - Use urfall processed data with fall/ and normal/ subdirs

# After training:
python scripts/test_fall_detection.py --video path/to/fall_test.mp4
```

---

## 6. Augmentation Strategy

Augmentation is applied only during training. The `FeatureWindowDataset` with `augment=True` applies `DEFAULT_TRAIN_AUGMENTATION` from `ml/training/data_augmentation.py`:

| Augmentation | Rate | Effect |
|-------------|------|--------|
| Gaussian noise on keypoints | Always | Simulates sub-pixel noise |
| Horizontal flip | 50% | Doubles dataset, handles both orientations |
| Temporal jitter | 30% | Simulates variable playback speed |
| Motion feature scaling | Always | Simulates different camera heights |
| Keypoint occlusion (0–4 kp) | Always | Simulates partial body visibility |

**Do not apply**: temporal reversal for violent/distress class (semantics change; falling down ≠ getting up).

---

## 7. Class Imbalance Handling

Normal activity always dominates. Without correction, models learn to always predict "normal."

Two mechanisms applied simultaneously:

**1. WeightedRandomSampler** — oversamples minority classes each epoch:
```python
class_weights = dataset.get_class_weights()  # inverse frequency
sampler = WeightedRandomSampler(sample_weights, len(dataset))
```

**2. Weighted CrossEntropyLoss** — larger gradient for minority class errors:
```python
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

This combination prevents the model from predicting only "normal."

---

## 8. Export to ONNX

After training, export for cross-platform deployment:

```bash
python -m ml.export.export_onnx \
    --checkpoint ml/checkpoints/temporal_v1.pt \
    --output ml/checkpoints/temporal_v1.onnx \
    --validate

# Validation output:
# ✓ ONNX model structure valid
# Max output difference PT vs ONNX: 2.34e-07
# ✓ Outputs match within tolerance
# ✓ ONNX model size: 1.24 MB
```

---

## 9. TensorRT Export (Jetson only)

Run on the Jetson device itself — TensorRT engines are hardware-specific:

```bash
# On Jetson:
python -m ml.export.export_tensorrt \
    --onnx ml/checkpoints/temporal_v1.onnx \
    --output ml/checkpoints/temporal_v1.trt \
    --fp16

# Then set in .env:
# INFERENCE__USE_TENSORRT=true

# Benchmark:
python scripts/benchmark_inference.py --frames 200 --resolution 480p
```

---

## 10. Dataset Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Staged/acted violence | Poor generalization to real incidents | Add custom CCTV data; validate on site |
| Low-light bias | Model fails at night | Include low-light augmentation; test across lighting |
| Camera angle bias | Fails on top-down or side-view | Collect data from multiple angles |
| Label ambiguity in UCF-Crime | Noisy labels degrade training | Filter manually; use only clearly labeled clips |
| Normal class dominance | Model biased to predict normal | Use WeightedRandomSampler + weighted loss |
| Data leakage (video → windows) | Inflated val metrics | Split by video, not by window |

**Most common training mistake**: Splitting windows randomly. Two windows from the same video (1 in train, 1 in val) create data leakage. Always split by video first, then extract windows from the respective split videos.
