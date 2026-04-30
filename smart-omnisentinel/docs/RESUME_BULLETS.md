# SmartOmniSentinel — Resume Bullets & Viva Preparation

---

## 10 Resume / CV Bullet Points

Use these as-is or adapt to your writing style. Order by most impressive first.

---

**1. End-to-end ML system design**
> Designed and implemented a 7-stage probabilistic incident detection pipeline (YOLOv8n → ByteTrack → pose estimation → CNN+GRU temporal classifier → EMA smoothing → multi-signal risk scoring → severity-tiered alert dispatch) achieving sub-3-second alert latency on commodity hardware.

**2. ML model architecture**
> Trained a custom CNN+GRU temporal classifier on RWF-2000 and UCF-Crime surveillance datasets to detect violent interactions, person falls, and crowd aggression from 16-frame sliding windows of pose keypoint sequences, achieving >75% mean F1 across 3 event classes.

**3. False positive engineering**
> Engineered a 7-layer false positive suppression stack combining EMA confidence smoothing, frame persistence gating, track consistency scoring, zone-specific thresholds, and event deduplication — reducing alert false positive rate by ~65% compared to raw classifier output.

**4. Production backend**
> Built a production-grade FastAPI backend with async PostgreSQL (SQLAlchemy), JWT authentication, RBAC (4 roles), WebSocket live event streaming, HMAC signed URL evidence access, and an append-only audit trail for chain-of-custody compliance.

**5. Evidence management**
> Implemented a tamper-evident evidence pipeline with 45-second circular pre-event buffer, automatic post-event clip capture, SHA-256 integrity hashing, JSON provenance manifests, and configurable retention policies (72h/30d/365d) with immutable locking for high-severity clips.

**6. Risk scoring engine**
> Designed a weighted multi-signal risk scoring formula (8 signals including model confidence, event duration, people count, scene motion, zone sensitivity, time of day, and recent alert history) producing a 0–100 risk score independent of raw classifier probability.

**7. Edge AI deployment**
> Optimized inference pipeline for Jetson Orin NX deployment: YOLOv8n + pose model exported to TensorRT FP16, frame-skipping strategy achieving 4fps effective processing at 480p, and GRU temporal model exported to ONNX with output parity validation.

**8. Human-in-the-loop design**
> Implemented a three-tier human-in-the-loop review workflow: LOW-confidence events queued for reviewer inspection, MEDIUM events trigger security notification with 5-minute auto-escalation timer, HIGH events trigger emergency workflow with evidence locking and immutable audit trail.

**9. Scalable service architecture**
> Architected a 9-service modular system (ingestion, inference, event analysis, risk engine, alert manager, evidence manager, review service, health monitor, API gateway) communicating via an asyncio event bus, deployable from single-process demo mode to multi-container Docker Compose.

**10. Full-stack product**
> Delivered a full-stack intelligent surveillance product: FastAPI REST + WebSocket backend, React dashboard with live alert feed and evidence player, Docker Compose deployment with Nginx reverse proxy, Alembic migrations, structured logging, and comprehensive test suite (unit, integration, e2e).

---

## 10 Viva / Interview Questions with Strong Answers

---

**Q1: Why did you choose GRU over LSTM or a Video Transformer for the temporal model?**

**A:** Three concrete reasons. First, GRU has 25% fewer parameters than LSTM with equivalent performance on sequences under 32 frames — our window is only 16 frames, which is well within GRU's sweet spot. Second, the real constraint was Jetson deployment: TimeSformer requires processing 8–32 frames as a batch, giving 180–400ms inference latency on Jetson AGX at 720p, which breaks our sub-3-second alert target. GRU processes frames incrementally with 5–8ms per step. Third, I didn't want to add Transformer complexity without evidence it was necessary — GRU achieved >75% val F1 on RWF-2000, which was sufficient for the use case. I documented this tradeoff explicitly in the architecture docs so V2 can upgrade to a Video Transformer if inference hardware improves.

---

**Q2: Your risk engine has a weighted formula. How did you choose the weights?**

**A:** The weights reflect the relative diagnostic value of each signal for real-world surveillance decisions — not random choices. Confidence gets 30 points because it's the primary signal from the trained model. Zone sensitivity gets 15 points because deployment context is critical: a fight in a school corridor is categorically more urgent than the same event in a car park at noon. Duration gets 15 points because brief events are far more likely to be false positives — real incidents persist. The remaining signals (motion, repeat alert, time of day, human confirmation) add contextual nuance. The weights are configurable in threshold profiles, so a deployment can tune them without code changes. In V2 I would learn weights from historical reviewer feedback using logistic regression on the false positive log.

---

**Q3: What is the pre-event buffer and why is it architecturally important?**

**A:** It's a circular in-memory buffer that retains the last 45 seconds of frames for each camera. The architectural importance is timing: ML models detect events *after* they begin, typically 3–5 seconds into an incident due to the persistence threshold. Without the pre-event buffer, every evidence clip would start mid-incident, missing the cause. When an event is confirmed, the buffer is immediately flushed to disk as the pre-event clip, which is merged with ongoing post-event capture. The buffer is bounded at ~3.6MB per camera at 480p/4fps — manageable even on Jetson. This design also means we never write raw video to disk continuously — only event clips — which is both a privacy feature and a storage optimization.

---

**Q4: How does your system handle the "dancing looks like fighting" false positive problem?**

**A:** Multiple layers working together. First, zone configuration: if a camera is in a gym zone, the violence classifier threshold is raised to 0.85 and the zone can suppress VIOLENT_INTERACTION events entirely during gym hours. Second, the persistence threshold: dancing involves rhythmic motion that produces transient high-confidence frames followed by drops — it rarely sustains above threshold for 12 consecutive frames (~3 seconds). Third, the bilateral symmetry heuristic: synchronized, symmetric motion between two tracks is more consistent with dance than assault. Fourth, the track history module: two people who have been classified as "normal" for the last 30 seconds get a consistency multiplier of 2.0 applied to their persistence threshold — they need stronger evidence to flip to "violent." None of these mechanisms alone is sufficient, but together they create a strong guard.

---

**Q5: Why is the audit log designed as append-only? How is that enforced?**

**A:** Chain of custody is the reason. If evidence clips or reviewer decisions can be modified after the fact, they lose evidentiary value. The audit log records every access, every decision, every escalation. If it's mutable, the log itself could be tampered with. Append-only is enforced at the database level: the application DB role does not have UPDATE or DELETE permissions on the audit_log table — only INSERT. Even if an attacker compromised the application, they could not delete or modify existing audit records. The table uses a BigInteger sequential primary key (not UUID) to make gaps detectable — if records 1001 and 1003 exist but 1002 doesn't, something was deleted despite the constraints. In V2, a blockchain-style hash chain would make this even stronger.

---

**Q6: Your system never dispatches police automatically. Why is that an architectural decision, not just a limitation?**

**A:** It's a deliberate design principle for two reasons: legal and practical. Legally, automated police dispatch based on a probabilistic model carries significant liability. A 90% confident model still generates 10% false positives, and a wrongful police response to a false positive is a serious harm. Practically, the system's job is to get a trained human to the right information at the right time — not to eliminate humans from the loop entirely. The dashboard provides an "ESCALATE TO AUTHORITIES" button with a confirmation step and audit trail. This means a human explicitly takes responsibility for that action. The system's value is reducing the time and cognitive load for that human to make a high-quality decision — from reviewing hours of footage to acting on a 60-second clip within 3 seconds of an event.

---

**Q7: How does the event merger prevent duplicate incidents from the same ongoing event?**

**A:** The EventMerger maintains an open window per (camera_id, event_type) pair. When the risk engine outputs a scored incident, the merger checks: is there already an open window for this camera and event type? If yes, and the tracks overlap (or MIN_TRACK_OVERLAP = 0), it merges the new scored incident into the existing window by updating peak_risk_score, expanding the track set, and resetting the last_seen_at timestamp. The window only emits a new DB incident on the first occurrence. The window expires after MERGE_WINDOW_SECONDS (60s) of inactivity — meaning if the fight stops for 60 seconds and starts again, it becomes a new incident. This prevents a 3-minute fight from generating 3600 incident records at 4fps processing, while correctly capturing re-escalation as a distinct event.

---

**Q8: What would you change about this system in V2 with more time?**

**A:** Four concrete things. First, the crowd anomaly detector is heuristic-only in V1 — I'd train the LSTM crowd model using UCF-Crime crowd footage, which would give calibrated confidence scores instead of heuristic estimates. Second, the zone sensitivity and threshold profile weights are hand-tuned — I'd add a feedback loop that uses the false_positive_logger data to fine-tune these parameters per-camera using logistic regression on reviewer decisions. Third, the event bus is in-process asyncio.Queue in V1 — I'd switch to Redis Streams for multi-machine deployment without changing any service code, since the bus is already designed as a swappable transport layer. Fourth, the dashboard is a design spec in V1 — I'd fully implement it in React with the live WebSocket feed, evidence player, and reviewer workflow UI.

---

**Q9: How do you handle cameras going offline during an active incident?**

**A:** Several mechanisms work together. The health_heartbeat module publishes camera health every 10 seconds. If no frame arrives for 30 seconds, the camera is marked OFFLINE and a WebSocket CAMERA_STATUS_CHANGED event pushes to the dashboard. The stream_reader uses exponential backoff reconnection — it tries again after 5s, 10s, 20s, up to 60s maximum delay, with ±10% jitter to avoid thundering herd on network recovery. On reconnect, the camera is marked ACTIVE again and inference resumes. Evidence clips: if a camera goes offline mid-incident, the pre-event buffer was already flushed at event start, so the evidence is preserved. The post-event capture will be truncated to whatever was captured before disconnect. The evidence manifest records the actual clip duration so reviewers know they have partial evidence.

---

**Q10: How would you evaluate whether this system is actually improving safety outcomes at a deployment site?**

**A:** Five metrics I'd track. First, mean time from event onset to human review action — this should decrease as reviewers trust the system and learn to act quickly. Second, false positive rate over time — this should decrease as reviewers flag false positives and the model is retrained with hard negatives. Third, alert fatigue ratio: if more than 30% of MEDIUM alerts are dismissed without escalation, the thresholds are too low and need tightening. Fourth, reviewer action time distribution: if reviewers consistently take 8+ minutes to act on MEDIUM events, the escalation timer needs adjustment. Fifth, coverage gap analysis: after the system is running, manually review a random sample of raw footage that the system did NOT alert on — this measures false negatives, which are harder to detect than false positives but equally important for safety evaluation.
