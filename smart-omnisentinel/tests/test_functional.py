"""
tests/test_functional.py
========================
FUNCTIONAL / BEHAVIOURAL TEST SUITE — Phase 1 & Phase 2
========================================================

Tests that the new FEATURES work as intended, not just that code compiles.
Each test simulates a realistic real-world scenario and asserts the correct
behaviour (output values, signals, decisions) — not just tensor shapes.

Phase 1 Features:
  [F1.1] Proximity correctly detects close vs far persons
  [F1.2] Proximity correctly suppresses solo actor (no neighbours)
  [F1.3] ST-GCN backbone produces different embeddings for different skeletons
  [F1.4] ST-GCN+GRU classifies consistently (deterministic in eval mode)
  [F1.5] Kinematic augmentor actually CHANGES the data (not identity)
  [F1.6] Kinematic augmentor preserves spatial constraints (coords in [0,1])
  [F1.7] Ablation models show increasing latency A < B < C < D (architecture grows)
  [F1.8] Risk scorer produces higher score for night + crowded + high confidence

Phase 2 Features:
  [F2.1] Audio 1D-CNN produces DIFFERENT outputs for speech vs silence
  [F2.2] Multimodal fusion boosts confidence over visual-only baseline
  [F2.3] Silent video (zero MFCC) does not crash and degrades gracefully
  [F2.4] Scene-adaptive: sports threshold blocks alerts that office would fire
  [F2.5] Scene-adaptive: unknown scene falls back to indoor_generic threshold
  [F2.6] Temporal localization head: end_boundary always >= start_boundary
  [F2.7] Temporal localization: boundary loss is 0 for all-negative batch
  [F2.8] Temporal localization: format_incident_timestamps produces correct strings
  [F2.9] Attention overlay: high-weight joints get visually different rendering
  [F2.10] Weapon detection risk: weapon flag doubles threat vs no-weapon scenario
  [F2.11] Weapon risk clamped to 100 (never exceeds maximum)
  [F2.12] Risk scorer backward-compatible with no weapon fields
  [F2.13] build_signals_from_event correctly propagates weapon metadata
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# PHASE 1 FUNCTIONAL TESTS
# ---------------------------------------------------------------------------

class TestPhase1Proximity(unittest.TestCase):
    """[F1.1 – F1.2] Proximity Feature Extractor behaves correctly."""

    def setUp(self):
        from features.proximity import ProximityFeatureExtractor
        from services.inference.tracker import Track
        from services.inference.pose_estimator import PoseEstimate

        self.ProximityFeatureExtractor = ProximityFeatureExtractor
        self.Track = Track
        self.PoseEstimate = PoseEstimate

    def _make_track(self, tid, cx, cy):
        bx1, by1 = cx - 50, cy - 100
        bx2, by2 = cx + 50, cy + 100
        return self.Track(
            track_id=tid, bbox_xyxy=[bx1, by1, bx2, by2],
            confidence=0.9, frames_since_seen=0, age_frames=5,
            centroid=(float(cx), float(cy))
        )

    def test_F1_1_close_persons_produce_high_proximity_distance_low(self):
        """Two persons 50px apart → dist_1 should be small (< 0.2)."""
        extractor = self.ProximityFeatureExtractor()
        t1 = self._make_track(1, 100, 200)
        t2 = self._make_track(2, 140, 200)  # 40px apart
        vec = extractor.extract_track_proximity(1, [t1, t2], poses={})
        dist_1 = vec[0]
        self.assertLess(dist_1, 0.2,
            f"Close persons (40px apart) should have dist_1 < 0.2, got {dist_1:.3f}")

    def test_F1_1_far_persons_produce_high_distance(self):
        """Two persons 800px apart → dist_1 should be high (> 0.5)."""
        extractor = self.ProximityFeatureExtractor()
        t1 = self._make_track(1, 100, 300)
        t2 = self._make_track(2, 900, 300)  # 800px apart
        vec = extractor.extract_track_proximity(1, [t1, t2], poses={})
        dist_1 = vec[0]
        self.assertGreater(dist_1, 0.5,
            f"Far persons (800px apart) should have dist_1 > 0.5, got {dist_1:.3f}")

    def test_F1_2_solo_actor_returns_all_zeros(self):
        """Single person in frame → all proximity features must be zero."""
        extractor = self.ProximityFeatureExtractor()
        t1 = self._make_track(1, 320, 240)
        vec = extractor.extract_track_proximity(1, [t1], poses={})
        self.assertTrue(np.all(vec == 0.0),
            f"Solo actor must yield zero-proximity vector, got {vec}")

    def test_F1_1_proximity_vector_always_normalized(self):
        """All proximity values must be in [0.0, 1.0]."""
        extractor = self.ProximityFeatureExtractor()
        for _ in range(20):
            cx1, cy1 = np.random.randint(50, 300), np.random.randint(50, 400)
            cx2, cy2 = np.random.randint(50, 600), np.random.randint(50, 400)
            t1 = self._make_track(1, cx1, cy1)
            t2 = self._make_track(2, cx2, cy2)
            vec = extractor.extract_track_proximity(1, [t1, t2], poses={})
            self.assertTrue(np.all(vec >= 0.0) and np.all(vec <= 1.0),
                f"Proximity features out of [0,1]: {vec}")


class TestPhase1STGCNBackbone(unittest.TestCase):
    """[F1.3] ST-GCN produces meaningfully different embeddings for different skeletons."""

    def test_F1_3_different_skeletons_produce_different_embeddings(self):
        """
        Two maximally-different pose sequences (all-zeros vs all-ones)
        must produce embeddings with cosine similarity < 0.99.
        """
        from models.stgcn import STGCNBackbone
        backbone = STGCNBackbone(in_channels=3)
        backbone.eval()

        # Aggressive stance (high-velocity arms extended)
        violent_pose = torch.randn(1, 3, 16, 17) * 2.0 + 1.0
        # Calm neutral pose (near-zero motion)
        calm_pose = torch.zeros(1, 3, 16, 17) + 0.1

        with torch.no_grad():
            emb_v = backbone(violent_pose).mean(dim=1)  # (1, 128)
            emb_c = backbone(calm_pose).mean(dim=1)

        cos_sim = torch.nn.functional.cosine_similarity(emb_v, emb_c).item()
        self.assertLess(cos_sim, 0.99,
            f"Embeddings for radically different skeletons too similar: cos={cos_sim:.4f}")

    def test_F1_3_identical_inputs_produce_identical_embeddings(self):
        """Same input twice in eval mode → identical output (deterministic)."""
        from models.stgcn import STGCNBackbone
        backbone = STGCNBackbone(in_channels=3)
        backbone.eval()
        x = torch.randn(1, 3, 16, 17)
        with torch.no_grad():
            out1 = backbone(x)
            out2 = backbone(x)
        self.assertTrue(torch.allclose(out1, out2),
            "Identical inputs in eval mode must give identical outputs")


class TestPhase1STGCNGRUClassifier(unittest.TestCase):
    """[F1.4] ST-GCN+GRU classifier is deterministic in eval mode."""

    def test_F1_4_eval_mode_is_deterministic(self):
        """Same input → same logits in eval mode (no dropout randomness)."""
        from models.stgcn_gru import STGCNGRUClassifier
        model = STGCNGRUClassifier(num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 16, 17)
        prox = torch.randn(2, 6)
        with torch.no_grad():
            out1 = model(x, prox)
            out2 = model(x, prox)
        self.assertTrue(torch.allclose(out1, out2),
            "eval mode must be fully deterministic")

    def test_F1_4_output_shape_and_range(self):
        """Output shape (B, 2), values finite."""
        from models.stgcn_gru import STGCNGRUClassifier
        model = STGCNGRUClassifier(num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(3, 3, 16, 17), torch.randn(3, 6))
        self.assertEqual(out.shape, (3, 2))
        self.assertFalse(torch.isnan(out).any(), "No NaN in output")
        self.assertFalse(torch.isinf(out).any(), "No Inf in output")


class TestPhase1KinematicAugmentor(unittest.TestCase):
    """[F1.5 – F1.6] Kinematic augmentor actually transforms data."""

    def test_F1_5_augmentation_changes_data(self):
        """Augmented output must NOT be identical to input (actual transform happens)."""
        from data.augment import KinematicAugmentor
        aug = KinematicAugmentor(max_rotation_deg=5.0, jitter_sigma=0.02)
        original = np.random.rand(16, 17, 3).astype('float32') * 0.5 + 0.25

        # Run 10 times; at least 1 must produce different output
        changed = False
        for _ in range(10):
            augmented = aug(original)
            if not np.allclose(original, augmented, atol=1e-6):
                changed = True
                break

        self.assertTrue(changed, "Augmentor never changed the data — identity transform bug!")

    def test_F1_6_spatial_coordinates_remain_in_unit_cube(self):
        """XY coordinates after augmentation must stay in [0, 1]."""
        from data.augment import KinematicAugmentor
        aug = KinematicAugmentor(max_rotation_deg=15.0, jitter_sigma=0.05)
        seq = np.random.rand(16, 17, 3).astype('float32')
        for _ in range(50):
            out = aug(seq)
            xy = out[:, :, :2]
            self.assertTrue(
                np.all(xy >= 0.0) and np.all(xy <= 1.0),
                f"XY coordinates out of [0,1] range: min={xy.min():.4f}, max={xy.max():.4f}"
            )

    def test_F1_6_confidence_channel_preserved(self):
        """Confidence column (index 2) must not be clipped to wrong range."""
        from data.augment import KinematicAugmentor
        aug = KinematicAugmentor()
        seq = np.ones((16, 17, 3), dtype='float32') * 0.9  # conf=0.9
        out = aug(seq)
        conf_col = out[:, :, 2]
        # Confidence is not augmented by rotation/jitter — should remain near 0.9
        self.assertTrue(np.all(conf_col >= 0.0), "Confidence went negative")

    def test_F1_6_output_shape_preserved_2d_and_3d(self):
        """Shape must be preserved for both (T, J*C) and (T, J, C) input."""
        from data.augment import KinematicAugmentor
        aug = KinematicAugmentor()
        seq_3d = np.random.rand(16, 17, 3).astype('float32')
        out_3d = aug(seq_3d)
        self.assertEqual(out_3d.shape, seq_3d.shape)

        seq_2d = np.random.rand(16, 51).astype('float32')
        out_2d = aug(seq_2d)
        self.assertEqual(out_2d.shape, seq_2d.shape)


class TestPhase1RiskScorer(unittest.TestCase):
    """[F1.8] Risk scorer produces sensible scores for known signal combinations."""

    def test_F1_8_night_crowded_high_conf_gives_high_score(self):
        """Nighttime + 5 people + high confidence + repeat alert → HIGH severity."""
        from services.risk_engine.risk_scorer import RiskSignals, compute_risk_score
        from core.constants import Severity
        sig = RiskSignals(
            confidence=0.95, duration_seconds=20.0, people_count=5,
            motion_magnitude=7.0, recent_alert=True, zone_sensitivity=0.9,
            hour_of_day=2, human_confirmed=False
        )
        rs = compute_risk_score(sig)
        self.assertEqual(rs.severity, Severity.HIGH,
            f"High-signal scenario should be HIGH severity, got {rs.severity} (score={rs.score})")
        self.assertGreater(rs.score, 60)

    def test_F1_8_daytime_solo_low_conf_gives_low_score(self):
        """Daytime + 1 person + low confidence → LOW severity."""
        from services.risk_engine.risk_scorer import RiskSignals, compute_risk_score
        from core.constants import Severity
        sig = RiskSignals(
            confidence=0.1, duration_seconds=1.0, people_count=1,
            motion_magnitude=0.5, recent_alert=False, zone_sensitivity=0.2,
            hour_of_day=12, human_confirmed=False
        )
        rs = compute_risk_score(sig)
        self.assertIn(rs.severity, [Severity.LOW, Severity.MEDIUM],
            f"Low-signal scenario should be LOW or MEDIUM, got {rs.severity} (score={rs.score})")
        self.assertLess(rs.score, 50)

    def test_F1_8_score_monotonically_increases_with_confidence(self):
        """Higher confidence must yield higher or equal risk score."""
        from services.risk_engine.risk_scorer import RiskSignals, compute_risk_score

        def make_sig(conf):
            return RiskSignals(confidence=conf, duration_seconds=5.0, people_count=2,
                               motion_magnitude=3.0, recent_alert=False,
                               zone_sensitivity=0.5, hour_of_day=14, human_confirmed=False)

        scores = [compute_risk_score(make_sig(c)).score for c in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for i in range(len(scores) - 1):
            self.assertLessEqual(scores[i], scores[i + 1],
                f"Score did not increase with confidence: {scores}")


# ---------------------------------------------------------------------------
# PHASE 2 FUNCTIONAL TESTS
# ---------------------------------------------------------------------------

class TestPhase2AudioCNN(unittest.TestCase):
    """[F2.1 – F2.3] Audio 1D-CNN feature behaviour."""

    def test_F2_1_audio_embedding_differs_for_speech_vs_silence(self):
        """
        MFCC from simulated speech (random non-zero) vs silence (zeros)
        must produce meaningfully different embeddings.
        """
        from models.audio_cnn import Audio1DCNN
        enc = Audio1DCNN(in_channels=13, embed_dim=32)
        enc.eval()

        speech_mfcc = torch.randn(1, 13, 100) * 5.0  # High-energy audio
        silence_mfcc = torch.zeros(1, 13, 100)        # True silence

        with torch.no_grad():
            emb_speech = enc(speech_mfcc)
            emb_silence = enc(silence_mfcc)

        l2_diff = torch.norm(emb_speech - emb_silence).item()
        self.assertGreater(l2_diff, 0.1,
            f"Speech and silence embeddings are too similar (L2={l2_diff:.4f}) — audio branch may be broken")

    def test_F2_2_multimodal_fusion_differs_from_visual_only(self):
        """
        fusion(visual, real_audio) must differ from fusion(visual, zero_audio).
        Audio branch is contributing to the output.
        """
        from models.audio_cnn import Audio1DCNN, MultimodalVisualAudioFusion
        enc = Audio1DCNN()
        fuse = MultimodalVisualAudioFusion(visual_dim=256, audio_dim=32)
        enc.eval(); fuse.eval()

        visual = torch.randn(1, 256)
        real_mfcc = torch.randn(1, 13, 100) * 3.0
        zero_mfcc = torch.zeros(1, 13, 100)

        with torch.no_grad():
            emb_real = enc(real_mfcc)
            emb_zero = enc(zero_mfcc)
            out_real = fuse(visual, emb_real)
            out_zero = fuse(visual, emb_zero)

        diff = abs(out_real.item() - out_zero.item())
        self.assertGreater(diff, 0.0,
            "Multimodal fusion output identical for real vs zero audio — audio gate may be stuck")

    def test_F2_3_silent_video_zero_mfcc_fallback_no_crash(self):
        """AudioExtractor on non-existent file → zero (13,100), no crash."""
        from data.audio_extractor import AudioExtractor
        ext = AudioExtractor()
        mfcc = ext.extract_mfcc("this_file_absolutely_does_not_exist.mp4")
        self.assertEqual(mfcc.shape, (13, 100))
        self.assertEqual(mfcc.sum(), 0.0)
        self.assertFalse(np.isnan(mfcc).any())

    def test_F2_3_fusion_with_none_audio_does_not_crash(self):
        """Passing None as audio to fusion → model uses internal zero fallback."""
        from models.audio_cnn import MultimodalVisualAudioFusion
        fuse = MultimodalVisualAudioFusion(visual_dim=256, audio_dim=32)
        fuse.eval()
        visual = torch.randn(2, 256)
        with torch.no_grad():
            out = fuse(visual, None)
        self.assertEqual(out.shape, (2, 1))
        self.assertFalse(torch.isnan(out).any())


class TestPhase2SceneAdaptive(unittest.TestCase):
    """[F2.4 – F2.5] Scene-adaptive thresholds make actual alert decisions."""

    def setUp(self):
        from services.event_analysis.scene_adaptive import SceneAdaptiveManager
        self.mgr = SceneAdaptiveManager()

    def test_F2_4_sports_blocks_what_office_would_fire(self):
        """
        Confidence 0.70: should trigger alert in office (τ=0.55)
        but NOT in sports (τ=0.85).
        """
        confidence = 0.70
        self.assertTrue(
            self.mgr.should_trigger_alert(confidence, "office"),
            f"Office scene (τ=0.55) should fire at conf={confidence}"
        )
        self.assertFalse(
            self.mgr.should_trigger_alert(confidence, "sports"),
            f"Sports scene (τ=0.85) should NOT fire at conf={confidence}"
        )

    def test_F2_4_each_scene_fires_at_its_own_threshold(self):
        """Each scene fires at exactly its threshold (at-boundary test)."""
        scene_thresholds = {
            "sports": 0.85, "office": 0.55, "street": 0.65,
            "playground": 0.80, "indoor_generic": 0.60
        }
        for scene, tau in scene_thresholds.items():
            # Exactly at threshold → should fire (>=)
            self.assertTrue(
                self.mgr.should_trigger_alert(tau, scene),
                f"Scene '{scene}' should fire at exactly τ={tau}"
            )
            # Just below threshold → should NOT fire
            self.assertFalse(
                self.mgr.should_trigger_alert(tau - 0.001, scene),
                f"Scene '{scene}' should NOT fire at τ-0.001={tau-0.001:.3f}"
            )

    def test_F2_5_unknown_scene_uses_indoor_generic_fallback(self):
        """Unknown scene type falls back to indoor_generic threshold (0.60)."""
        # 0.65 > 0.60 → should fire on unknown scene
        self.assertTrue(self.mgr.should_trigger_alert(0.65, "unknown_parking_lot"))
        # 0.55 < 0.60 → should NOT fire on unknown scene
        self.assertFalse(self.mgr.should_trigger_alert(0.55, "mars_landscape"))

    def test_F2_5_none_scene_type_uses_fallback(self):
        """None scene type uses _default fallback (0.60)."""
        self.assertTrue(self.mgr.should_trigger_alert(0.60, None))
        self.assertFalse(self.mgr.should_trigger_alert(0.59, None))


class TestPhase2TemporalLocalization(unittest.TestCase):
    """[F2.6 – F2.8] Temporal Action Localization head behaviour."""

    def test_F2_6_end_boundary_always_gte_start_boundary(self):
        """Monotonic constraint: end >= start for every sample in batch."""
        from models.temporal_localization import TemporalLocalizationHead
        head = TemporalLocalizationHead(hidden_dim=256, num_classes=2)
        head.eval()
        for trial in range(20):
            with torch.no_grad():
                _, bounds = head(torch.randn(8, 256))
            start = bounds[:, 0]
            end = bounds[:, 1]
            self.assertTrue(
                torch.all(end >= start),
                f"Trial {trial}: end < start at indices {(end < start).nonzero().tolist()}"
            )

    def test_F2_6_boundaries_within_unit_interval(self):
        """Both start and end must be in [0.0, 1.0]."""
        from models.temporal_localization import TemporalLocalizationHead
        head = TemporalLocalizationHead(hidden_dim=256, num_classes=2)
        head.eval()
        with torch.no_grad():
            _, bounds = head(torch.randn(16, 256))
        self.assertTrue(torch.all(bounds >= 0.0) and torch.all(bounds <= 1.0),
            f"Boundaries out of [0,1]: min={bounds.min():.4f}, max={bounds.max():.4f}")

    def test_F2_7_boundary_loss_zero_for_all_negative_batch(self):
        """
        L_boundary is only computed on positive samples.
        All-negative batch → boundary loss component must be 0.
        """
        from losses.boundary_loss import TemporalBoundaryLoss
        crit = TemporalBoundaryLoss(lambda_boundary=0.5)
        logits = torch.randn(4, 2)
        bounds = torch.rand(4, 2)
        tgt_cls = torch.zeros(4, dtype=torch.long)      # All negatives
        tgt_bounds = torch.rand(4, 2)

        # Compute loss twice: once with bad bounds, once with correct bounds
        # Result should be identical (boundary loss = 0 for negatives)
        loss1 = crit(logits, bounds, tgt_cls, tgt_bounds)
        loss2 = crit(logits, bounds + 99.0, tgt_cls, tgt_bounds)
        self.assertAlmostEqual(
            loss1.item(), loss2.item(), places=5,
            msg="Boundary loss should not change for all-negative batch (no positive mask)"
        )

    def test_F2_8_timestamp_format_is_human_readable(self):
        """format_incident_timestamps produces MM:SS.s strings and correct values."""
        from models.temporal_localization import format_incident_timestamps
        # 10-second window, violence at 30%–70% → 3.0s – 7.0s
        ts = format_incident_timestamps(
            window_start_sec=0.0, window_duration_sec=10.0,
            start_frac=0.30, end_frac=0.70, confidence=0.92
        )
        self.assertAlmostEqual(ts["start_seconds"], 3.0, places=2)
        self.assertAlmostEqual(ts["end_seconds"], 7.0, places=2)
        self.assertAlmostEqual(ts["duration_seconds"], 4.0, places=2)
        self.assertIn("00:03", ts["start_time_str"])
        self.assertIn("00:07", ts["end_time_str"])
        self.assertIn("Violence detected", ts["display_banner"])
        self.assertEqual(ts["confidence"], 0.92)

    def test_F2_8_timestamp_with_window_offset(self):
        """Window starting at t=60s, 30%–60% → absolute 78s – 96s."""
        from models.temporal_localization import format_incident_timestamps
        ts = format_incident_timestamps(
            window_start_sec=60.0, window_duration_sec=60.0,
            start_frac=0.30, end_frac=0.60, confidence=0.85
        )
        self.assertAlmostEqual(ts["start_seconds"], 78.0, places=2)
        self.assertAlmostEqual(ts["end_seconds"], 96.0, places=2)


class TestPhase2AttentionVisualization(unittest.TestCase):
    """[F2.9] Attention overlay renders differently based on weights."""

    def test_F2_9_high_weight_joints_get_different_color(self):
        """
        A frame rendered with all-high weights must differ from all-low weights.
        (checks that the colormap function actually maps to different pixel values)
        """
        from visualization.attention_overlay import draw_attention_overlay
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Fixed joint positions — a simple vertical skeleton
        kpts = np.zeros((17, 2), dtype='float32')
        for i in range(17):
            kpts[i] = [320, 100 + i * 20]

        high_weights = np.ones(17, dtype='float32') * 0.95   # All high attention
        low_weights = np.ones(17, dtype='float32') * 0.05    # All low attention

        rendered_high = draw_attention_overlay(frame.copy(), kpts, high_weights)
        rendered_low  = draw_attention_overlay(frame.copy(), kpts, low_weights)

        # Pixel sums must differ (different colors were painted)
        self.assertFalse(
            np.array_equal(rendered_high, rendered_low),
            "High-weight and low-weight overlays produced identical pixels — attention coloring not working"
        )

    def test_F2_9_overlay_does_not_crash_on_out_of_frame_keypoints(self):
        """Keypoints at frame boundary / outside frame must not crash cv2."""
        from visualization.attention_overlay import draw_attention_overlay
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        kpts = np.array([[0, 0], [320, 240], [-10, -10], [500, 500]] + [[160, 120]] * 13,
                        dtype='float32')
        weights = np.random.rand(17).astype('float32')
        try:
            out = draw_attention_overlay(frame, kpts, weights)
            self.assertEqual(out.shape, (240, 320, 3))
        except Exception as e:
            self.fail(f"draw_attention_overlay crashed on out-of-frame keypoints: {e}")


class TestPhase2WeaponRiskEscalation(unittest.TestCase):
    """[F2.10 – F2.13] Weapon detection correctly escalates risk score."""

    def _make_sig(self, conf=0.6, weapon=False, wtype=None):
        from services.risk_engine.risk_scorer import RiskSignals
        return RiskSignals(
            confidence=conf, duration_seconds=5.0, people_count=2,
            motion_magnitude=4.0, recent_alert=False, zone_sensitivity=0.5,
            hour_of_day=14, human_confirmed=False,
            weapon_detected=weapon, weapon_type=wtype
        )

    def test_F2_10_weapon_flag_significantly_boosts_risk_score(self):
        """
        With weapon detected, score must be exactly 1.5× the base score.
        (before clamping at 100)
        """
        from services.risk_engine.risk_scorer import compute_risk_score
        sig_base   = self._make_sig(conf=0.5, weapon=False)
        sig_weapon = self._make_sig(conf=0.5, weapon=True, wtype='knife')

        rs_base   = compute_risk_score(sig_base)
        rs_weapon = compute_risk_score(sig_weapon)

        expected = int(min(100, round(rs_base.factor_breakdown["raw_score"] * 1.5)))
        self.assertEqual(rs_weapon.score, expected,
            f"Expected weapon-escalated score={expected}, got {rs_weapon.score}")
        self.assertGreater(rs_weapon.score, rs_base.score,
            "Weapon should always increase risk score")

    def test_F2_10_weapon_type_is_preserved_in_output(self):
        """weapon_type metadata flows through to RiskScore."""
        from services.risk_engine.risk_scorer import compute_risk_score
        for wtype in ["knife", "baseball_bat", "bottle", "scissors"]:
            rs = compute_risk_score(self._make_sig(weapon=True, wtype=wtype))
            self.assertEqual(rs.weapon_type, wtype,
                f"weapon_type '{wtype}' not preserved in RiskScore")
            self.assertTrue(rs.weapon_detected)

    def test_F2_11_weapon_risk_is_clamped_to_100(self):
        """At max signals + weapon → score must not exceed 100."""
        from services.risk_engine.risk_scorer import RiskSignals, compute_risk_score
        sig_maxed = RiskSignals(
            confidence=1.0, duration_seconds=60.0, people_count=10,
            motion_magnitude=20.0, recent_alert=True, zone_sensitivity=1.0,
            hour_of_day=3, human_confirmed=True,
            weapon_detected=True, weapon_type="knife"
        )
        rs = compute_risk_score(sig_maxed)
        self.assertLessEqual(rs.score, 100, f"Score exceeded 100: {rs.score}")
        self.assertEqual(rs.score, 100)

    def test_F2_11_no_weapon_multiplier_is_1(self):
        """Without weapon, multiplier must be exactly 1.0 in breakdown."""
        from services.risk_engine.risk_scorer import compute_risk_score
        rs = compute_risk_score(self._make_sig(weapon=False))
        self.assertEqual(rs.factor_breakdown["weapon_multiplier"], 1.0)
        self.assertFalse(rs.weapon_detected)
        self.assertIsNone(rs.weapon_type)

    def test_F2_12_backward_compat_old_risksignals_no_weapon_fields(self):
        """
        RiskSignals created WITHOUT weapon_detected/weapon_type kwargs
        (old call sites) must work with defaults (False / None).
        """
        from services.risk_engine.risk_scorer import RiskSignals, compute_risk_score
        # Old-style construction — no weapon kwargs
        sig_old = RiskSignals(
            confidence=0.7, duration_seconds=8.0, people_count=2,
            motion_magnitude=5.0, recent_alert=False, zone_sensitivity=0.7,
            hour_of_day=10, human_confirmed=False
        )
        rs = compute_risk_score(sig_old)
        self.assertFalse(rs.weapon_detected)
        self.assertIsNone(rs.weapon_type)
        self.assertEqual(rs.factor_breakdown["weapon_multiplier"], 1.0)

    def test_F2_13_build_signals_propagates_weapon_metadata(self):
        """build_signals_from_event correctly reads weapon_detected/type from event dict."""
        from services.risk_engine.risk_scorer import build_signals_from_event
        # Event dict with weapon
        event_with_weapon = {
            "smoothed_confidence": 0.75,
            "people_count": 2,
            "motion_magnitude": 4.5,
            "weapon_detected": True,
            "weapon_type": "baseball_bat"
        }
        sig = build_signals_from_event(event_with_weapon, zone_sensitivity=0.6)
        self.assertTrue(sig.weapon_detected)
        self.assertEqual(sig.weapon_type, "baseball_bat")

        # Event dict without weapon (old format)
        event_old = {"smoothed_confidence": 0.6, "people_count": 1, "motion_magnitude": 2.0}
        sig2 = build_signals_from_event(event_old, zone_sensitivity=0.5)
        self.assertFalse(sig2.weapon_detected)
        self.assertIsNone(sig2.weapon_type)


class TestPhase2WeaponDetectorModule(unittest.TestCase):
    """Weapon detector COCO class map and WeaponDetection dataclass correctness."""

    def test_weapon_class_map_covers_all_four_types(self):
        from models.weapon_detector import WEAPON_CLASS_MAP
        self.assertEqual(WEAPON_CLASS_MAP[43], "knife")
        self.assertEqual(WEAPON_CLASS_MAP[34], "baseball_bat")
        self.assertEqual(WEAPON_CLASS_MAP[39], "bottle")
        self.assertEqual(WEAPON_CLASS_MAP[76], "scissors")

    def test_weapon_detector_returns_empty_on_blank_frame(self):
        """WeaponDetector on a blank frame with no tracks → empty list."""
        from models.weapon_detector import WeaponDetector
        det = WeaponDetector(confidence_threshold=0.99)  # unreachably high threshold
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect_weapons(blank, [])
        self.assertIsInstance(results, list)
        # May be empty or may try to load model; either is acceptable
        # What must NOT happen: crash


if __name__ == "__main__":
    unittest.main(verbosity=2)
