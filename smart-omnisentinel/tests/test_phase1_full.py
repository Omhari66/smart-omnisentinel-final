"""
tests/test_phase1_full.py
-------------------------
Comprehensive Automated Test & Verification Suite for Phase 1.

Validates all 7 Days:
  [Day 1] ONNX/TensorRT model exports & latency breakdown
  [Day 2] 6-dim inter-person proximity feature extraction
  [Day 3] ST-GCN graph backbone & spatial pooling
  [Day 4] ST-GCN + GRU hybrid fusion classifier
  [Day 5] Kinematic augmentations (rotation, jitter, drop, speed)
  [Day 6] 4-variant ablation progression (A -> B -> C -> D)
  [Day 7] Cross-dataset evaluation & domain shift suite
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase1Full(unittest.TestCase):

    def test_day1_exports_and_latency_docs(self):
        """Verify Day 1 deliverables: ONNX models and latency breakdown documentation."""
        pose_onnx = PROJECT_ROOT / "models" / "yolov8n-pose.onnx"
        detector_onnx = PROJECT_ROOT / "models" / "yolov8n.onnx"
        latency_doc = PROJECT_ROOT / "docs" / "latency_breakdown.md"
        profile_script = PROJECT_ROOT / "scripts" / "profile_latency.py"
        export_script = PROJECT_ROOT / "scripts" / "export_yolo_tensorrt.py"

        self.assertTrue(pose_onnx.exists(), "models/yolov8n-pose.onnx missing")
        self.assertTrue(detector_onnx.exists(), "models/yolov8n.onnx missing")
        self.assertTrue(latency_doc.exists(), "docs/latency_breakdown.md missing")
        self.assertTrue(profile_script.exists(), "scripts/profile_latency.py missing")
        self.assertTrue(export_script.exists(), "scripts/export_yolo_tensorrt.py missing")

        # Verify latency doc content
        with open(latency_doc, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Latency Breakdown Table", content)
            self.assertIn("ST-GCN", content)

    def test_day2_proximity_features(self):
        """Verify Day 2 deliverables: 6-dim proximity vector and graceful fallbacks."""
        from features.proximity import ProximityFeatureExtractor
        from services.inference.pose_estimator import PoseEstimate
        from services.inference.tracker import Track

        extractor = ProximityFeatureExtractor()

        # Dummy tracks and poses
        t1 = Track(track_id=1, bbox_xyxy=[100, 100, 200, 300], confidence=0.9, frames_since_seen=0, age_frames=5, centroid=(150.0, 200.0))
        t2 = Track(track_id=2, bbox_xyxy=[180, 100, 280, 300], confidence=0.85, frames_since_seen=0, age_frames=5, centroid=(230.0, 200.0))

        kp1 = np.random.rand(17, 2).astype(np.float32)
        kp2 = np.random.rand(17, 2).astype(np.float32)
        pose1 = PoseEstimate(track_id=1, keypoints=kp1, keypoint_conf=np.ones(17), is_valid=True, bbox_xyxy=[100, 100, 200, 300])
        pose2 = PoseEstimate(track_id=2, keypoints=kp2, keypoint_conf=np.ones(17), is_valid=True, bbox_xyxy=[180, 100, 280, 300])

        vec = extractor.extract_track_proximity(
            target_track_id=1,
            tracks=[t1, t2],
            poses={1: pose1, 2: pose2},
        )
        self.assertEqual(vec.shape, (6,), "Proximity vector must be 6-dimensional")
        self.assertTrue(np.all(vec >= 0.0) and np.all(vec <= 1.0), "Proximity features must be normalized in [0, 1]")

        # Solo track test (single person exercising alone)
        solo_vec = extractor.extract_track_proximity(target_track_id=1, tracks=[t1], poses={1: pose1})
        self.assertTrue(np.all(solo_vec == 0.0), "Single person track must yield zero-proximity prior")

    def test_day3_stgcn_backbone(self):
        """Verify Day 3 deliverables: ST-GCN graph backbone topology and forward pass."""
        from models.stgcn import STGCNBackbone

        backbone = STGCNBackbone(in_channels=3)
        backbone.eval()

        dummy_x = torch.randn(2, 3, 16, 17)
        with torch.no_grad():
            emb = backbone(dummy_x)

        self.assertEqual(emb.shape, (2, 16, 128), "ST-GCN backbone output must be (batch, time=16, embed=128)")

    def test_day4_stgcn_gru_fusion(self):
        """Verify Day 4 deliverables: ST-GCN + GRU hybrid fusion model."""
        from models.stgcn_gru import STGCNGRUClassifier

        model = STGCNGRUClassifier(in_channels=3, proximity_dim=6, hidden_size=256, num_layers=2)
        model.eval()

        dummy_x = torch.randn(2, 3, 16, 17)
        dummy_prox = torch.randn(2, 16, 6)
        with torch.no_grad():
            logits = model(dummy_x, dummy_prox)

        self.assertEqual(logits.shape, (2, 2), "ST-GCN+GRU logits must be (batch, 2)")

        # Verify training script exists
        train_script = PROJECT_ROOT / "train_stgcn_gru.py"
        self.assertTrue(train_script.exists(), "train_stgcn_gru.py missing")

    def test_day5_kinematic_augmentor(self):
        """Verify Day 5 deliverables: Kinematic augmentation pipeline."""
        from data.augment import KinematicAugmentor

        augmentor = KinematicAugmentor()
        orig = np.random.rand(16, 51).astype(np.float32)
        augmented = augmentor(orig)

        self.assertEqual(augmented.shape, orig.shape, "Augmentation must preserve sequence dimensions")
        self.assertFalse(np.isnan(augmented).any(), "Augmentation must not produce NaN values")

    def test_day6_ablation_study(self):
        """Verify Day 6 deliverables: Ablation table and benchmark suite."""
        ablation_doc = PROJECT_ROOT / "experiments" / "ablation_table.md"
        ablation_script = PROJECT_ROOT / "experiments" / "run_ablation_benchmarks.py"

        self.assertTrue(ablation_doc.exists(), "experiments/ablation_table.md missing")
        self.assertTrue(ablation_script.exists(), "experiments/run_ablation_benchmarks.py missing")

        with open(ablation_doc, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Variant A", content)
            self.assertIn("Variant B", content)
            self.assertIn("Variant C", content)
            self.assertIn("Variant D", content)

    def test_day7_cross_dataset_generalization(self):
        """Verify Day 7 deliverables: Cross-dataset benchmark report and evaluation suite."""
        cross_doc = PROJECT_ROOT / "experiments" / "cross_dataset.md"
        eval_script = PROJECT_ROOT / "ml" / "datasets" / "cross_dataset_eval.py"

        self.assertTrue(cross_doc.exists(), "experiments/cross_dataset.md missing")
        self.assertTrue(eval_script.exists(), "ml/datasets/cross_dataset_eval.py missing")

        with open(cross_doc, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("RWF-2000", content)
            self.assertIn("RLVS", content)
            self.assertIn("Violent Flows", content)


if __name__ == "__main__":
    unittest.main()

