"""tests/unit/test_confidence_smoother.py — EMA smoother and persistence checker."""

import pytest
from services.event_analysis.event_analyzer import SmoothingState, EventAnalyzer


class TestSmoothingState:

    def test_ema_convergence(self):
        state = SmoothingState()
        alpha = 0.35
        threshold = 0.50
        # Feed high confidence 20 times
        for _ in range(20):
            state.update(0.90, alpha, threshold)
        assert state.smoothed_confidence > 0.70

    def test_persistence_increments_above_threshold(self):
        """
        EMA starts at 0.0. With alpha=0.35 and input=0.90, the smoothed value
        only crosses the 0.50 threshold after ~3 updates. The test feeds 10
        frames and checks that consecutive_frames is > 0 (not exactly N).
        """
        state = SmoothingState()
        for _ in range(10):
            state.update(0.90, 0.35, 0.50)
        # Should have accumulated several consecutive frames above threshold
        assert state.consecutive_frames > 0

    def test_persistence_resets_below_threshold(self):
        """
        Feed enough high-confidence frames to build up consecutive_frames,
        then feed low-confidence to verify the counter decrements.
        """
        state = SmoothingState()
        # Build up count: after many high-confidence updates, smoothed >> 0.50
        for _ in range(20):
            state.update(0.90, 0.35, 0.50)
        frames_after_high = state.consecutive_frames
        assert frames_after_high > 0

        # Feed low confidence — each update decrements consecutive_frames by 1
        # (implementation: max(0, count - 1) when below threshold)
        for _ in range(frames_after_high + 5):
            state.update(0.10, 0.35, 0.50)

        assert state.consecutive_frames < frames_after_high

    def test_decay_reduces_confidence(self):
        state = SmoothingState()
        for _ in range(10):
            state.update(0.85, 0.35, 0.50)
        before = state.smoothed_confidence
        state.decay(0.35)
        assert state.smoothed_confidence < before

    def test_event_start_timestamp_set_after_threshold_crossed(self):
        """
        EMA starts at 0.0 so the first few updates may be below threshold.
        We feed many frames at high confidence until smoothed crosses 0.50,
        then assert the timestamp is set.
        """
        state = SmoothingState()
        assert state.event_start_timestamp is None

        # Feed enough frames for EMA to converge above 0.50
        for _ in range(15):
            state.update(0.90, 0.35, 0.50)

        # After 15 frames at 0.90 with alpha=0.35, smoothed >> 0.50
        assert state.event_start_timestamp is not None

    def test_event_start_timestamp_cleared_on_reset(self):
        state = SmoothingState()
        # Build up high confidence
        for _ in range(20):
            state.update(0.90, 0.35, 0.50)
        assert state.event_start_timestamp is not None

        # Feed very low values repeatedly — fast alpha decays smoothed to 0
        for _ in range(30):
            state.update(0.00, 0.90, 0.50)  # High alpha, zero input → fast decay
        assert state.consecutive_frames == 0
        assert state.event_start_timestamp is None


class TestEventAnalyzerCooldown:

    def test_cooldown_blocks_events(self):
        analyzer = EventAnalyzer(cooldown_seconds=300.0)
        analyzer.set_cooldown("cam-001", "VIOLENT_INTERACTION")
        cooldown = analyzer._cooldowns["cam-001"]["VIOLENT_INTERACTION"]
        assert cooldown.is_active()

    def test_cooldown_resets_smoothing(self):
        analyzer = EventAnalyzer(cooldown_seconds=300.0)
        state = analyzer._smoothing["cam-001"]["VIOLENT_INTERACTION"]
        state.smoothed_confidence = 0.95
        state.consecutive_frames = 20

        analyzer.set_cooldown("cam-001", "VIOLENT_INTERACTION")

        # After cooldown, state should be reset
        new_state = analyzer._smoothing["cam-001"]["VIOLENT_INTERACTION"]
        assert new_state.smoothed_confidence == 0.0
        assert new_state.consecutive_frames == 0
