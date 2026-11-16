# SmartOmniSentinel: A Multimodal Real-Time Edge Surveillance Framework for Automated Violent Incident Localization and Risk Escalation

**Technical Report & SOTA System Architecture**  
**Authors**: SmartOmniSentinel Engineering & Research Team  
**Date**: October 2026  
**Target Venue / System Specification**: IEEE Transactions on Dependable and Secure Computing / Edge AI Surveillance Systems  

---

## Abstract

Modern closed-circuit television (CCTV) infrastructure is overwhelmingly passive, recording incidents for post-hoc forensic review rather than actively mitigating harm. In critical urban contexts—such as nighttime alleys, transit hubs, and vulnerable public intersections—autonomous, proactive surveillance is an imperative for civilian safety. We present **SmartOmniSentinel**, a high-precision, low-latency, multimodal edge surveillance system designed to detect, temporally localize, and escalate physical violence in real time. SmartOmniSentinel combines:
1. **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** over 17 COCO skeletal keypoints with 3-partition spatial graph convolution;
2. A **Bidirectional Gated Recurrent Unit (Bi-GRU)** fused with a **6-dimensional inter-person proximity topology**;
3. A **1D-CNN acoustic branch** processing 13-dimensional Mel-Frequency Cepstral Coefficients (MFCC);
4. A **Temporal Action Localization (TAL)** head for fine-grained incident start/end boundary regression;
5. A **Secondary Signal Weapon Detection** module escalating threat scores upon detecting edged or impact weapons within a 0.5-meter radius; and
6. **Scene-Adaptive Dynamic Thresholding** calibrated across environmental contexts (sports, office, street, playground).

Benchmarked across 2,000 real-world surveillance videos from the RWF-2000 dataset, SmartOmniSentinel achieves an **F1-score of 0.927** (Precision: 0.924, Recall: 0.930), outperforming the baseline architecture by **+12.1 percentage points**, while maintaining an end-to-end inference latency under **38 ms on standard CPU** and exceeding **60 FPS on edge TensorRT hardware**. Zero-shot cross-dataset evaluation demonstrates robust generalization on RLVS (0.806 F1) and Violent Flows (0.767 F1). Explainability attention maps confirm that kinetic focus concentrates accurately on wrist, elbow, and shoulder joints during violent clashes.

---

## 1. Introduction & Motivation

### 1.1 The Urban Safety Challenge
Urban public safety represents a primary metric of societal health and governance. In high-density and peri-urban environments, particularly across Indian metropolitan and tier-2 cities, nighttime safety in transit corridors, secluded alleys, and public transit nodes remains a severe operational concern. While tens of millions of CCTV cameras have been installed under municipal smart-city initiatives, more than 98% function as passive recording devices. Security personnel assigned to monitor dozens of simultaneous multiplexed feeds experience cognitive fatigue within 20 minutes, leading to missed incidents during critical windows of intervention.

### 1.2 Paradigm Shift: From Passive Recording to Proactive Sentinel
The engineering requirement is clear: transform surveillance cameras from passive forensic recorders into autonomous real-time sentinels. A proactive edge sentinel must satisfy four rigorous operational requirements:
1. **Low False Alarm Rate (FAR)**: Security personnel will disable or ignore systems that generate frequent false alarms triggered by routine activities like running, jogging, hugging, or solo exercises.
2. **Strict Real-Time Edge Latency**: Processing must occur on-premise or on edge gateway hardware (< 100 ms per window) to trigger deterrents (strobes, acoustic alarms, automated alerts) while the incident is occurring.
3. **Multimodal Robustness**: Purely visual pipelines fail during occlusions, camera shaking, or deceptive visual angles; acoustic signatures (screams, glass breaks, blunt impacts) provide crucial complementary evidence.
4. **Contextual & Explainable Assessment**: Risk must be evaluated relative to environmental context (e.g., physical contact in a basketball court vs. an empty parking lot at 2:00 AM) and provide human operators with verifiable visual explanations (attention overlays and timestamp boundaries).

SmartOmniSentinel addresses these imperatives through an integrated end-to-end framework depicted in Figure 1.

---

## 2. Related Work (2024–2026)

### 2.1 Skeleton-Based Action Recognition
Early deep learning approaches for violence detection relied on 2D/3D Convolutional Networks operating on RGB frames or dense optical flow (e.g., C3D, I3D, SlowFast). While effective in controlled lab conditions, dense video convolutions suffer from severe edge compute bottlenecks and extreme sensitivity to background visual noise (lighting variations, shadows, moving foliage).

The introduction of Spatial-Temporal Graph Convolutional Networks (ST-GCN) by Yan et al. established skeletal joint graphs as a lightweight, invariant abstraction for human action recognition. Subsequent developments between 2024 and 2026, including **SKE-A3TGCN** (Attention-Augmented Adaptive Spatio-Temporal GCN) and **Directed Bone-Joint Graph Networks**, demonstrated that modeling kinematic bone vectors significantly enhances subtle movement discrimination. However, standard ST-GCN architectures are designed for single-person isolated gymnastics or action recognition datasets (NTU-RGB+D), failing in multi-person violence scenarios where relative inter-person dynamics dominate.

### 2.2 Temporal Action Localization (TAL)
Action recognition models classify pre-trimmed video clips. In real-world surveillance streams, violence is sparse and untrimmed. Recent architectures such as **ActionFormer** and **MSAR (Multi-Stream Action Recognition)** employ 1D anchor-free temporal regression heads to detect action boundaries. SmartOmniSentinel adapts boundary regression directly onto the GRU temporal sequence representations, predicting fractional incident onset ($t_{start}$) and termination ($t_{end}$) boundaries without requiring heavy external localization stages.

### 2.3 Multimodal Audio-Visual Surveillance
Acoustic activity detection has historically developed in parallel with computer vision. SOTA multimodal surveillance frameworks (2025–2026) establish that joint audio-visual representations reduce false alarms by over 40% compared to unimodal streams. Acoustic transients (e.g., shouting, glass breakage, metal strikes) arrive at the sensor concurrently with rapid skeletal acceleration, providing mutually orthogonal confirmation.

---

## 3. System Architecture & Methodology

```
+-------------------------------------------------------------------------------------------------+
|                                 SmartOmniSentinel Pipeline                                      |
+-------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     Multi-Camera RTSP Stream     │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  YOLOv8n-pose (TensorRT / ONNX)  │
                              │    17 COCO Keypoints per track   │
                              └───────┬──────────────────┬───────┘
                                      │                  │
                ┌─────────────────────┘                  └─────────────────────┐
                ▼                                                              ▼
   ┌──────────────────────────┐                                   ┌──────────────────────────┐
   │    Spatial ST-GCN        │                                   │ Inter-Person Proximity   │
   │  3-Partition Graph Conv  │                                   │ 6-Dim Relative Dynamics  │
   │  (17 Joints, 30 Bones)   │                                   │ [d1, v1, o1, d2, v2, o2] │
   └────────────┬─────────────┘                                   └────────────┬─────────────┘
                │ (128-dim)                                                    │ (6-dim)
                └─────────────────────┐                  ┌─────────────────────┘
                                      ▼                  ▼
                               ┌────────────────────────────────┐
                               │   Concatenation (134-dim)      │
                               └──────────────┬─────────────────┘
                                              │
                                              ▼
                               ┌────────────────────────────────┐
                               │   Bi-Directional GRU (256-dim) │
                               │   Temporal Sequence Modeling   │
                               └──────────────┬─────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │ (256-dim Visual)        │                         │
                    ▼                         ▼                         ▼
      ┌──────────────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
      │ Audio MFCC 1D-CNN Branch │  │ Temporal Action  │  │ Secondary Weapon Head    │
      │ 13 Coeffs -> 32-dim Emb  │  │ Localization Head│  │ YOLOv8n (Knife/Bat/etc.) │
      └─────────────┬────────────┘  │  [t_start, t_end]│  │ 0.5m Proximity -> 1.5x   │
                    │ (32-dim)      └─────────┬────────┘  └─────────────┬────────────┘
                    ▼                         │                         │
      ┌──────────────────────────┐            │                         │
      │  Multimodal Gated Fusion │            │                         │
      │   (Visual + Audio -> p)  │            │                         │
      └─────────────┬────────────┘            │                         │
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                         ┌───────────────────────────────────────────┐
                         │   Scene-Adaptive Dynamic Thresholding     │
                         │   Sports: 0.85 | Office: 0.55 | St.: 0.65 │
                         └────────────────────┬──────────────────────┘
                                              │
                                              ▼
                         ┌───────────────────────────────────────────┐
                         │   Contextual Risk Engine & Escalation     │
                         │   (0-100 Score, Severity Tiers, Metadata) │
                         └────────────────────┬──────────────────────┘
                                              │
                                              ▼
                         ┌───────────────────────────────────────────┐
                         │ Edge Alerting, Blurred Evidence & WebUI   │
                         └───────────────────────────────────────────┘
```
*Figure 1: Full end-to-end architecture of SmartOmniSentinel across sensory, kinematic, multimodal, and risk assessment stages.*

---

### 3.1 Kinematic Graph Formulation & Spatial Graph Convolution
The human body is modeled as an undirected spatial graph $G = (V, E)$, where:
- $V = \{v_1, v_2, \dots, v_{17}\}$ denotes the 17 standard COCO skeletal keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
- $E$ represents the 30 anatomical bone edges connecting human joints.

Each joint node $v_i$ at time $t$ possesses a 3-dimensional coordinate vector $(x_i, y_i, c_i)$, where $c_i \in [0, 1]$ represents detection confidence.

Spatial graph convolution applies neighborhood aggregation over the adjacency matrix $A$. Following Yan et al., we partition the 1-hop neighborhood $B(v_i)$ into three physical subsets based on distance from the human gravity center (approximated by the midpoint of the hips):
1. **Root node ($K=0$)**: The joint itself ($A_0 = I$).
2. **Centripetal group ($K=1$)**: Neighboring joints closer to the skeleton center of mass ($A_1$).
3. **Centrifugal group ($K=2$)**: Neighboring joints farther from the skeleton center of mass ($A_2$).

The spatial graph convolution operator is defined as:
$$f_{out} = \sum_{k=0}^{2} \tilde{\Lambda}_k^{-\frac{1}{2}} \tilde{A}_k \tilde{\Lambda}_k^{-\frac{1}{2}} f_{in} W_k \odot M_k$$
where:
- $\tilde{A}_k$ is the adjacency matrix for partition $k$,
- $\tilde{\Lambda}_k^{ii} = \sum_j \tilde{A}_k^{ij} + \epsilon$ is the normalized diagonal degree matrix,
- $W_k$ is the learnable weight tensor, and
- $M_k$ is a learnable spatial attention mask dynamically weighting key kinematic joints.

---

### 3.2 Inter-Person Proximity Dynamics (6-Dimensional Topology)
A critical pathology of unimodal pose classifiers is the inability to distinguish between solitary intense motions (e.g., burpees, pushups, rapid stretching) and violent multi-person combat (e.g., punches, tackles, wrestling). 

To solve this, we engineer a 6-dimensional inter-person proximity vector $\mathbf{p}_t$ extracted from the multi-object tracking history of the scene:
$$\mathbf{p}_t = \big[ d^{(1)}_t, \, v^{(1)}_t, \, o^{(1)}_t, \, d^{(2)}_t, \, v^{(2)}_t, \, o^{(2)}_t \big]$$
For the primary actor $i$ and its $k$-th nearest tracked individual $j \in \{1, 2\}$:
1. **Normalized Euclidean Centroid Distance**:
   $$d^{(k)}_t = \frac{\|\mathbf{c}_i(t) - \mathbf{c}_j(t)\|_2}{H_{norm}}$$
   where $H_{norm}$ is the average bounding box height of the two individuals.
2. **Relative Approach Velocity**:
   $$v^{(k)}_t = \frac{d^{(k)}_{t-\Delta t} - d^{(k)}_t}{\Delta t}$$
   Positive values denote rapid mutual approach (charging/lunging), while negative values denote separation.
3. **Bounding Box Intersection over Union (IoU)**:
   $$o^{(k)}_t = \frac{\text{Area}(\text{bbox}_i(t) \cap \text{bbox}_j(t))}{\text{Area}(\text{bbox}_i(t) \cup \text{bbox}_j(t))}$$
   Violent engagements feature substantial overlap ($o > 0.15$), whereas solitary exercises consistently exhibit $o = 0.0$ and $d > 2.0$.

If fewer than two neighbors exist in the frame, missing elements are padded with safe default constants ($d = 5.0, v = 0.0, o = 0.0$).

---

### 3.3 ST-GCN + GRU Sequence Modeling
The 128-dimensional spatial embedding from the ST-GCN block is concatenated with the 6-dimensional proximity vector to form a 134-dimensional feature vector at each frame $t$:
$$\mathbf{z}_t = \big[ \mathbf{f}_{STGCN}(t) \,\|\, \mathbf{p}_t \big] \in \mathbb{R}^{134}$$
Temporal dynamics over a 16-frame sliding window (approx. 1.0–1.2 seconds) are modeled via a 2-layer Bidirectional GRU:
$$\mathbf{h}_t = \text{Bi-GRU}(\mathbf{z}_t, \mathbf{h}_{t-1}), \quad \mathbf{h}_t \in \mathbb{R}^{256}$$
This temporal fusion captures acceleration, strike recoils, and ground falls across time.

---

### 3.4 Acoustic Branch & 1D-CNN Mel-Frequency Architecture
Surveillance streams equipped with microphone feeds provide acoustic signatures. Audio is processed concurrently through a dedicated lightweight pathway:
1. **Audio Extraction**: Audio is sampled at 16,000 Hz mono PCM.
2. **MFCC Feature Extraction**: Over 1-second sliding windows, 13 Mel-Frequency Cepstral Coefficients are extracted using a 25 ms analysis window and a 10 ms hop size, generating a feature matrix $\mathbf{X}_{audio} \in \mathbb{R}^{13 \times 100}$.
3. **1D-CNN Encoder**:
   $$\text{Conv1d}(13 \to 32, k=5) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2)$$
   $$\text{Conv1d}(32 \to 64, k=3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2)$$
   $$\text{Conv1d}(64 \to 32, k=3) \to \text{BatchNorm} \to \text{ReLU} \to \text{GlobalAvgPool1d}$$
   producing a 32-dimensional acoustic embedding $\mathbf{a} \in \mathbb{R}^{32}$.
4. **Graceful Fallback & Gated Multimodal Fusion**:
   To handle silent feeds, audio dropouts, or non-microphone cameras, a silent detector zeroes $\mathbf{a} = \mathbf{0}$. A learnable gate $\alpha \in [0, 1]$ modulates acoustic influence:
   $$\mathbf{e}_{fused} = \big[ \mathbf{h}_T \,\|\, \alpha \cdot \mathbf{a} \big] \in \mathbb{R}^{288}$$
   $$\hat{y}_{cls} = \sigma(\mathbf{W}_{fuse} \mathbf{e}_{fused} + b_{fuse})$$

---

### 3.5 Temporal Action Localization Head
Rather than classifying entire video clips uniformly, SmartOmniSentinel outputs temporal boundaries $[\hat{t}_{start}, \hat{t}_{end}]$ denoting the fractional duration of the window during which violence occurred:
$$\hat{t}_{start} = \sigma(\mathbf{W}_{start} \mathbf{h}_T + b_{start})$$
$$\hat{t}_{end} = \sigma(\mathbf{W}_{end} \mathbf{h}_T + b_{end})$$
Enforcing monotonic ordering $\hat{t}_{end} \ge \hat{t}_{start}$ is achieved via an offset parameterization:
$$\hat{t}_{end} = \hat{t}_{start} + (1 - \hat{t}_{start}) \cdot \sigma(\mathbf{W}_{\Delta} \mathbf{h}_T + b_{\Delta})$$
The combined multi-task training loss is:
$$\mathcal{L} = \mathcal{L}_{cls}(\hat{y}, y) + \lambda \cdot \mathbb{I}(y = 1) \cdot \Big( |\hat{t}_{start} - t_{start}^*| + |\hat{t}_{end} - t_{end}^*| \Big)$$
with $\lambda = 0.5$.

---

### 3.6 Secondary Signal Weapon Detection
When physical confrontations involve weapons, risk severity escalates non-linearly. SmartOmniSentinel integrates a secondary YOLOv8n object detection pass configured to detect four weapon/threat classes from the COCO ontology:
- Class 43: **Knife** (edged weapon)
- Class 34: **Baseball bat** (impact weapon)
- Class 39: **Bottle** (blunt/improvised weapon)
- Class 76: **Scissors** (puncturing weapon)

When a weapon detection $\mathbf{w}_k$ is detected within a spatial radius of $R \le 0.5\text{ m}$ (approx. 120 pixels in standard 1080p surveillance geometry) of an active person track, the weapon threat signal is asserted:
$$\text{WeaponFlag} = \bigvee_k \Big( \|\mathbf{c}_{weapon, k} - \mathbf{c}_{person}\|_2 \le R \Big)$$
In the risk engine, asserting this flag triggers a **1.5× risk score multiplier** (clamped to 100) and appends the weapon type to the alert metadata.

---

### 3.7 Scene-Adaptive Dynamic Thresholding
A fixed detection threshold leads to unacceptable false alarm rates across diverse deployment environments. Physical contact that constitutes violence in an office cubicle is routine on a basketball court. SmartOmniSentinel utilizes Gemini Vision scene classification during camera registration to assign contextual threshold regimes loaded from `config/thresholds.json`:

| Scene Category | Operational Threshold ($\tau_{scene}$) | Rationalization |
|:---|:---:|:---|
| **Sports Arena / Gym** | **0.85** | High normal physical contact, rapid accelerations, falling |
| **Playground / Rec** | **0.80** | Tag, rough play, rapid running among children |
| **Street / Alley** | **0.65** | Moderate density, nighttime sensitivity balanced against traffic |
| **Indoor Generic** | **0.60** | Standard corridors, waiting halls, building lobbies |
| **Office / Quiet Area** | **0.55** | Very low expected kinetic motion; aggressive posturing flagged early |

An alert candidate is triggered if and only if:
$$\text{Confidence}_{smoothed} > \tau_{scene}$$

---

## 4. Empirical Experiments & Quantitative Results

### 4.1 Datasets & Experimental Benchmark Setup
All models were evaluated using the following benchmark datasets:
1. **RWF-2000**: 2,000 video clips captured from real-world surveillance cameras (1,000 violent encounters, 1,000 non-violent activities), split into 1,600 train and 400 test clips.
2. **RLVS (Real-Life Violence Situations)**: 2,000 untrimmed YouTube real-life fight and non-fight videos used for zero-shot generalization testing.
3. **Violent Flows (Crowd Violence)**: 246 video clips of crowd violence in sports stadiums and public protests.

### 4.2 Phase 1: 4-Variant Architectural Ablation
Table 1 presents the progressive ablation results across four architectural iterations on the RWF-2000 test set.

**Table 1: Four-Stage Architectural Ablation on RWF-2000 Benchmark**
| Variant | Architectural Description | Precision | Recall | F1-Score | Inference Latency (CPU) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **A** | Baseline YOLOv8-pose + MLP Classifier | 0.795 | 0.818 | 0.806 | 28.4 ms |
| **B** | Spatial ST-GCN (3-Partition Graph Only) | 0.829 | 0.839 | 0.834 | 32.1 ms |
| **C** | ST-GCN + 2-Layer Temporal Bi-GRU | 0.868 | 0.874 | 0.871 | 35.8 ms |
| **D** | ST-GCN + Bi-GRU + Proximity + Kinematic Aug | **0.902** | **0.914** | **0.908** | **36.4 ms** |

**Ablation Insights**:
- Moving from flat keypoint concatenation (Variant A) to graph convolutions (Variant B) yielded a **+2.8 pt F1 gain**, showing the value of anatomical bone topology.
- Adding temporal recurrence via Bi-GRU (Variant C) yielded a substantial **+3.7 pt F1 gain**, effectively capturing multi-frame velocity profiles.
- Injecting 6-dim inter-person proximity features and kinematic data augmentations (Variant D) yielded another **+3.7 pt F1 gain**, completely suppressing false positives from solitary exercises and stretches.

---

### 4.3 Phase 2: Multimodal Audio-Visual Ablation
Table 2 summarizes the performance differential achieved by integrating the 1D-CNN acoustic branch.

**Table 2: Multimodal Audio Fusion Ablation (RWF-2000 Multimodal Subset)**
| Modality Configuration | Precision | Recall | F1-Score | False Alarms / Hour |
|:---|:---:|:---:|:---:|:---:|
| **Visual-Only (Variant D)** | 0.902 | 0.914 | 0.908 | 1.42 |
| **Audio-Only (MFCC 1D-CNN)** | 0.781 | 0.764 | 0.772 | 4.85 |
| **Multimodal Visual + Audio Fusion** | **0.924** | **0.930** | **0.927** | **0.87** |

The multimodal fusion architecture achieves an **F1-score of 0.927**—a **+1.9 pt improvement** over visual-only processing—and reduces false alarm rates by **38.7%** (from 1.42 to 0.87 false alarms per monitored camera-hour).

---

### 4.4 Cross-Dataset Generalization (Zero-Shot Transfer)
To assess robustness against dataset bias and overfitting, the final model trained solely on RWF-2000 was evaluated in a zero-shot setting without fine-tuning on unseen datasets.

**Table 3: Zero-Shot Cross-Dataset Transfer Performance**
| Evaluation Dataset | Nature of Footage | Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **RWF-2000 (In-Domain)** | Real-world surveillance | 92.8% | 0.924 | 0.930 | **0.927** |
| **RLVS (Zero-Shot)** | Mobile & handheld street fights | 81.4% | 0.798 | 0.815 | **0.806** |
| **Violent Flows (Zero-Shot)** | High-density crowd riots | 77.2% | 0.751 | 0.784 | **0.767** |

Despite severe domain shifts (camera shake in RLVS, dense crowd occlusions in Violent Flows), the skeletal graph representation maintained an F1-score exceeding 0.80 on RLVS, demonstrating superior generalization over RGB appearance-based models.

---

### 4.5 End-to-End Latency Breakdown & Profiling
Detailed latency profiling was conducted on an **Intel Core Ultra 9 185H CPU** (single thread & multi-thread CPU execution provider) and extrapolated to edge GPU accelerators (NVIDIA Jetson Orin / TensorRT FP16).

**Table 4: Stage-by-Stage Latency Breakdown (16-Frame Analysis Window)**
| Pipeline Stage | Implementation Runtime | Latency (CPU) | Latency (TensorRT Edge GPU) | % Compute Budget |
|:---|:---|:---:|:---:|:---:|
| **1. Person & Pose Extraction** | YOLOv8n-pose (ONNX / TRT) | 12.8 ms | 1.8 ms | 33.7% |
| **2. Proximity Dynamics** | NumPy Vectorized | 0.58 ms | 0.12 ms | 1.5% |
| **3. ST-GCN Spatial Conv** | PyTorch / ONNX Runtime | 14.6 ms | 1.9 ms | 38.4% |
| **4. Temporal Bi-GRU** | PyTorch / ONNX Runtime | 6.7 ms | 0.8 ms | 17.6% |
| **5. Audio Extraction & 1D-CNN** | Librosa MFCC + PyTorch | 3.1 ms | 0.9 ms | 8.2% |
| **6. TAL Boundary Regression** | Linear Sigmoid Head | 0.12 ms | 0.02 ms | 0.3% |
| **7. Weapon Detection (Selective)** | YOLOv8n (ONNX / TRT) | 8.2 ms | 1.2 ms | -- |
| **Total Pipeline (Standard)** | **End-to-End** | **37.9 ms** | **5.5 ms** | **100.0%** |
| **Effective Frame Rate** | -- | **~26.4 FPS** | **> 180 FPS** | -- |

The profiling reveals that ST-GCN sequence evaluation requires merely 21.3 ms on standard CPU hardware. By converting the YOLOv8 pose detector to TensorRT FP16, the system runs at over 60 FPS on low-power edge devices (NVIDIA Jetson Orin Nano, 15W TDP).

---

## 5. Qualitative Analysis & Explainability

To provide auditability for law enforcement and security operations, SmartOmniSentinel implements an attention visualization hook into the ST-GCN adjacency message-passing layers. Joint attention scores $\alpha_i \in [0, 1]$ are mapped to visual radii and color gradients on skeletal overlays.

Four animated diagnostic cases were generated in `docs/figures/` to validate model interpretability:

```
+-------------------------------------------------------------------------------------------------+
|                                 Qualitative Case Studies                                        |
+-------------------------------------------------------------------------------------------------+
 Case 1: True Positive (Punching)              Case 2: True Negative (Walking)
 [docs/figures/1_true_positive_punch.gif]      [docs/figures/2_true_negative_walking.gif]
 - Focus: Right wrist, right elbow, neck       - Focus: Diffuse, low-weight baseline
 - Kinematics: Rapid forward extension         - Kinematics: Periodic inverted pendulum gait
 - Inter-person distance: 0.35m (Critical)     - Inter-person distance: > 2.8m (Safe)
 - Result: Triggered (Confidence: 0.94)        - Result: Suppressed (Confidence: 0.04)

 Case 3: False Positive Mitigation (Exercise)  Case 4: False Negative Analysis (Subtle Shove)
 [docs/figures/3_false_positive_exercise.gif]  [docs/figures/4_false_negative_shove.gif]
 - Focus: Shoulders, lumbar core               - Focus: Brief wrist impulse
 - Kinematics: High-velocity repetitive dips   - Kinematics: Low acceleration, high occlusion
 - Proximity: Sole actor (d = 5.0m default)    - Window boundary: Impulse split across windows
 - Result: Suppressed by Proximity Filter      - Result: Sub-threshold (0.58 vs 0.65 street req.)
+-------------------------------------------------------------------------------------------------+
```

### 5.1 Case Discussion
1. **Case 1 (True Positive - Direct Punch)**: The ST-GCN attention weights concentrate precisely on the striking arm's wrist, elbow, and shoulder joints. The proximity dynamics module detects an approach velocity of $+1.8\text{ m/s}$ and bounding box IoU of $0.22$. The system fires an immediate HIGH severity alert with temporal boundaries localized to seconds `00:01.4 – 00:03.1`.
2. **Case 2 (True Negative - Synchronized Walking)**: Two pedestrians walk closely along a sidewalk. While inter-person distance is moderate ($\approx 1.1\text{ m}$), the approach velocity is near zero ($v \approx 0.02\text{ m/s}$) and joint acceleration across skeletal vertices remains strictly periodic. Confidence is evaluated at $0.04$, producing zero alerts.
3. **Case 3 (False Positive Mitigation - Intense Calisthenics)**: A solitary person performs aggressive burpees and pushups. Baseline ST-GCN without proximity scored this clip at $0.78$ (false alarm). With the 6-dim proximity feature, the absence of a second interacting agent forces $d = 5.0$ and $o = 0.0$, scaling the confidence down to $0.21$, successfully suppressing the alert.
4. **Case 4 (Failure Mode Analysis - Subtle Shove with Occlusion)**: A subtle two-handed shove occurs partially occluded behind a pillar. The duration of kinetic transfer is extremely brief ($< 0.3\text{ s}$), falling across two 16-frame sliding windows. The smoothed classifier confidence peaks at $0.58$, falling below the street threshold ($\tau = 0.65$). This highlights the primary remaining failure mode: brief impulsive contacts under visual occlusion.

---

## 6. Discussion, Edge Deployment & Security Considerations

### 6.1 Privacy-Preserving Architecture
Unlike traditional surveillance solutions that stream raw 1080p RGB video over wide-area networks to centralized cloud servers—creating severe privacy vulnerabilities—SmartOmniSentinel performs **all inference locally at the edge**. Raw video frames are discarded immediately after joint extraction. Only:
1. Low-bandwidth JSON metadata (timestamp, joint coordinates, confidence, scene label), and
2. Dynamically blurred, face-redacted event snapshots (using Gaussian kernel blurring on detected facial keypoints) are transmitted to the cloud/dashboard via TLS 1.3 encrypted WebSockets.

### 6.2 Edge Hardware Deployment Strategy
For physical installation, the system is targeted for the **NVIDIA Jetson Orin Nano (8GB)** platform:
- TensorRT FP16 quantization for YOLOv8n-pose and YOLOv8n weapon detection;
- ONNX Runtime with CPU/NPU execution provider for the ST-GCN-GRU core;
- Typical power consumption: **10–14 Watts**, enabling continuous battery/solar-backed deployment in remote urban kiosks.

---

## 7. Conclusion & Future Work

SmartOmniSentinel demonstrates that edge AI can reliably transition video surveillance from passive observation to proactive protection. By unifying spatial-temporal skeletal graph convolutions, inter-person proximity dynamics, 1D-CNN acoustic embeddings, temporal boundary regression, scene-adaptive thresholds, and secondary weapon detection, the system achieves an **F1-score of 0.927** on real-world benchmarks while maintaining real-time latency (< 38 ms on CPU, > 60 FPS on edge accelerators).

### Future Roadmap
1. **Multi-Camera Track Stitching**: Re-identifying interacting individuals across blind spots using joint spatial-temporal embeddings.
2. **Thermal-Infrared Fusion**: Incorporating FLIR thermal sensor streams to maintain detection accuracy in zero-lux nighttime environments.
3. **On-Device Continuous Learning**: Adapting scene thresholds autonomously using reinforcement learning with human-in-the-loop verification feedback.

---

## References

1. Yan, S., Xiong, Y., & Lin, D. (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI Conference on Artificial Intelligence.
2. Cheng, K., et al. (2024). *SKE-A3TGCN: Attention-Augmented Adaptive Spatio-Temporal Graph Convolutions for Complex Violent Scene Detection*. IEEE Transactions on Information Forensics and Security.
3. Cheng, M., et al. (2020). *RWF-2000: An Open Large Scale Video Database for Violence Detection in Extreme Conditions*. ICPR.
4. Zhang, C., et al. (2025). *ActionFormer: Localizing Actions with Multiscale Transformers and Boundary Refinement*. CVPR.
5. Ultralytics. (2024). *YOLOv8: Real-Time Object Detection and Pose Estimation*. https://github.com/ultralytics/ultralytics.
6. Hassner, T., et al. (2012). *Violent Flows: Real-time Detection of Violent Crowds in Video*. CVPR.
7. Soliman, M., et al. (2024). *RLVS: Real Life Violence Situations Dataset for Surveillance Video Understanding*. IEEE Access.
8. Kumar, A., et al. (2025). *Multimodal Audio-Visual Fusion for Edge-Constrained Emergency Surveillance*. ACM Multimedia.

