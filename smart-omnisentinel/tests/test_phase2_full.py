"""
tests/test_phase2_full.py
-------------------------
Comprehensive Automated Test & Verification Suite for Phase 2.

Validates all 5 Days of Phase 2:
  [Day 8]  Audio Branch (MFCC -> 1D-CNN) & Multimodal Visual-Audio Gated Fusion
  [Day 9]  Scene-Adaptive Dynamic Thresholding & JSON Configuration
  [Day 10] Temporal Action Localization (Start/End Boundary Regression Head & Loss)
  [Day 11] Attention Visualization (ST-GCN Joint Weights & 4 Diagnostic GIFs)
  [Day 12] Secondary Signal Weapon Detection (1.5x Escalation) & Publication Report
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase2Full(unittest.TestCase):

    def test_day8_audio_branch_and_multimodal_fusion(self):
        """Verify Day 8: Audio 1D-CNN, silent fallback, and multimodal visual-audio fusion."""
        from models.audio_cnn import Audio1DCNN, MultimodalVisualAudioFusion
        from data.audio_extractor import AudioExtractor

        audio_model_file = PROJECT_ROOT / "models" / "audio_cnn.py"
        audio_data_file = PROJECT_ROOT / "data" / "audio_extractor.py"
        audio_ablation_file = PROJECT_ROOT / "experiments" / "audio_ablation.md"

        self.assertTrue(audio_model_file.exists(), "models/audio_cnn.py missing")
        self.assertTrue(audio_data_file.exists(), "data/audio_extractor.py missing")
        self.assertTrue(audio_ablation_file.exists(), "experiments/audio_ablation.md missing")

        # Verify Audio 1D-CNN forward pass: (batch, 13, 100) -> (batch, 32)
        audio_cnn = Audio1DCNN(in_channels=13, embed_dim=32)
        dummy_mfcc = torch.randn(2, 13, 100)
        audio_emb = audio_cnn(dummy_mfcc)
        self.assertEqual(audio_emb.shape, (2, 32), "Audio1DCNN output must be (batch, 32)")

        # Verify Multimodal Visual+Audio Fusion forward pass: (batch, 256) + (batch, 32) -> (batch, 1)
        fusion_module = MultimodalVisualAudioFusion(visual_dim=256, audio_dim=32)
        dummy_visual = torch.randn(2, 256)
        cls_logits = fusion_module(dummy_visual, audio_emb)
        self.assertEqual(cls_logits.shape, (2, 1), "MultimodalVisualAudioFusion output must be (batch, 1)")

        # Verify graceful fallback with zero audio vector (silent video)
        zero_audio = torch.zeros(2, 32)
        cls_logits_silent = fusion_module(dummy_visual, zero_audio)
        self.assertEqual(cls_logits_silent.shape, (2, 1))
        self.assertFalse(torch.isnan(cls_logits_silent).any())

        # Verify AudioExtractor silent fallback dummy generation
        extractor = AudioExtractor()
        silent_mfcc = extractor.extract_mfcc("non_existent_or_silent.mp4")
        self.assertEqual(silent_mfcc.shape, (13, 100), "Silent fallback MFCC must be (13, 100)")
        self.assertEqual(float(np.sum(silent_mfcc)), 0.0, "Silent fallback MFCC must be all zeros")

    def test_day9_scene_adaptive_thresholding(self):
        """Verify Day 9: Scene-adaptive threshold configuration and gating manager."""
        from services.event_analysis.scene_adaptive import SceneAdaptiveManager

        config_file = PROJECT_ROOT / "config" / "thresholds.json"
        self.assertTrue(config_file.exists(), "config/thresholds.json missing")

        with open(config_file, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

        required_scenes = ["sports", "office", "street", "playground", "indoor_generic"]
        for scene in required_scenes:
            self.assertIn(scene, thresholds, f"Scene '{scene}' missing from config/thresholds.json")
            self.assertGreater(thresholds[scene], 0.0)
            self.assertLessEqual(thresholds[scene], 1.0)

        # Verify threshold manager evaluation
        manager = SceneAdaptiveManager(config_path=str(config_file))
        self.assertEqual(manager.get_threshold("sports"), 0.85)
        self.assertEqual(manager.get_threshold("office"), 0.55)
        self.assertEqual(manager.get_threshold("unknown_scene"), 0.60, "Fallback threshold must be 0.60")

        # Gating check: 0.70 confidence triggers alert in office (0.55) but suppressed in sports (0.85)
        self.assertTrue(manager.should_trigger_alert(0.70, "office"))
        self.assertFalse(manager.should_trigger_alert(0.70, "sports"))

    def test_day10_temporal_action_localization(self):
        """Verify Day 10: Temporal action localization head and boundary loss."""
        from models.temporal_localization import (
            TemporalLocalizationHead,
            format_incident_timestamps,
        )
        from losses.boundary_loss import TemporalBoundaryLoss

        tal_head = TemporalLocalizationHead(hidden_dim=256, num_classes=2)
        dummy_h = torch.randn(4, 256)
        logits, boundaries = tal_head(dummy_h)

        self.assertEqual(logits.shape, (4, 2))
        self.assertEqual(boundaries.shape, (4, 2))

        start_b = boundaries[:, 0]
        end_b = boundaries[:, 1]

        # Check boundary bounds [0, 1] and monotonic constraint (end >= start)
        self.assertTrue(torch.all(start_b >= 0.0) and torch.all(start_b <= 1.0))
        self.assertTrue(torch.all(end_b >= 0.0) and torch.all(end_b <= 1.0))
        self.assertTrue(torch.all(end_b >= start_b), "End boundary must be >= start boundary")

        # Verify loss calculation
        criterion = TemporalBoundaryLoss(lambda_boundary=0.5)
        target_cls = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        target_boundaries = torch.tensor([[0.2, 0.8], [0.0, 0.0], [0.4, 0.9], [0.0, 0.0]])

        total_loss = criterion(logits, boundaries, target_cls, target_boundaries)
        self.assertFalse(torch.isnan(total_loss))
        self.assertGreater(float(total_loss.item()), 0.0)

        # Verify human-readable timestamp formatting
        formatted = format_incident_timestamps(
            window_start_sec=0.0,
            window_duration_sec=10.0,
            start_frac=0.25,
            end_frac=0.75,
            confidence=0.88,
        )
        self.assertIn("00:02.5", formatted["start_time_str"])
        self.assertIn("00:07.5", formatted["end_time_str"])

    def test_day11_attention_visualization_and_gif_artifacts(self):
        """Verify Day 11: Attention overlay module and 4 diagnostic animated GIFs."""
        from visualization.attention_overlay import draw_attention_overlay

        # Verify dummy frame overlay generation
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_kpts = np.ones((17, 2), dtype=np.float32)
        dummy_kpts[:, 0] = np.linspace(100, 300, 17)
        dummy_kpts[:, 1] = np.linspace(100, 300, 17)
        dummy_weights = np.random.uniform(0.1, 0.9, size=17).astype(np.float32)

        rendered = draw_attention_overlay(dummy_frame, dummy_kpts, dummy_weights)
        self.assertEqual(rendered.shape, (480, 640, 3))

        # Verify 4 GIF deliverables exist and have non-zero size
        expected_gifs = [
            "1_true_positive_punch.gif",
            "2_true_negative_walking.gif",
            "3_false_positive_exercise.gif",
            "4_false_negative_shove.gif",
        ]
        figures_dir = PROJECT_ROOT / "docs" / "figures"
        for gif_name in expected_gifs:
            gif_path = figures_dir / gif_name
            self.assertTrue(gif_path.exists(), f"Figure '{gif_name}' missing from docs/figures")
            self.assertGreater(gif_path.stat().st_size, 10000, f"Figure '{gif_name}' is too small/corrupt")

    def test_day12_weapon_detection_and_risk_escalation(self):
        """Verify Day 12: Secondary weapon detection and 1.5x risk score escalation."""
        from models.weapon_detector import WeaponDetector, WEAPON_CLASS_MAP
        from services.risk_engine.risk_scorer import RiskSignals, compute_risk_score

        # Check weapon classes
        self.assertIn(43, WEAPON_CLASS_MAP)  # knife
        self.assertIn(34, WEAPON_CLASS_MAP)  # baseball bat
        self.assertIn(39, WEAPON_CLASS_MAP)  # bottle
        self.assertIn(76, WEAPON_CLASS_MAP)  # scissors

        # Verify Risk Scoring without weapon
        sig_baseline = RiskSignals(
            confidence=0.6,
            duration_seconds=6.0,
            people_count=2,
            motion_magnitude=4.0,
            recent_alert=False,
            zone_sensitivity=0.5,
            hour_of_day=14,
            human_confirmed=False,
            weapon_detected=False,
            weapon_type=None,
        )
        score_base = compute_risk_score(sig_baseline)

        # Verify Risk Scoring with weapon
        sig_weapon = RiskSignals(
            confidence=0.6,
            duration_seconds=6.0,
            people_count=2,
            motion_magnitude=4.0,
            recent_alert=False,
            zone_sensitivity=0.5,
            hour_of_day=14,
            human_confirmed=False,
            weapon_detected=True,
            weapon_type="knife",
        )
        score_weapon = compute_risk_score(sig_weapon)

        self.assertTrue(score_weapon.weapon_detected)
        self.assertEqual(score_weapon.weapon_type, "knife")
        # Multiplier check: 1.5x escalation
        expected_score = int(min(100, round(score_base.factor_breakdown["raw_score"] * 1.5)))
        self.assertEqual(score_weapon.score, expected_score)
        self.assertEqual(score_weapon.factor_breakdown["weapon_multiplier"], 1.5)

    def test_day12_publication_report_deliverable(self):
        """Verify Day 12: Final comprehensive publication-quality report."""
        report_file = PROJECT_ROOT / "report" / "final_report.md"
        self.assertTrue(report_file.exists(), "report/final_report.md missing")

        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        required_sections = [
            "Abstract",
            "1. Introduction & Motivation",
            "2. Related Work",
            "3. System Architecture & Methodology",
            "4. Empirical Experiments & Quantitative Results",
            "5. Qualitative Analysis & Explainability",
            "6. Discussion, Edge Deployment & Security Considerations",
            "7. Conclusion & Future Work",
            "References",
        ]
        for section in required_sections:
            self.assertIn(section, content, f"Report missing section: '{section}'")

        # Verify key empirical numbers and tables are documented
        self.assertIn("0.927", content)  # Final Multimodal F1
        self.assertIn("RWF-2000", content)
        self.assertIn("RLVS", content)
        self.assertIn("Violent Flows", content)


if __name__ == "__main__":
    unittest.main()

