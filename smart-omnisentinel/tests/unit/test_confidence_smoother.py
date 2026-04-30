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
        state = SmoothingState()
        for _ in range(5):
            state.update(0.90, 0.35, 0.50)
        assert state.consecutive_frames == 5

    def test_persistence_resets_below_threshold(self):
        state = SmoothingState()
        for _ in range(10):
            state.update(0.90, 0.35, 0.50)
        assert state.consecutive_frames == 10

        # Push low confidence — should decay frames
        for _ in range(6):
            state.update(0.10, 0.35, 0.50)
        assert state.consecutive_frames < 5

    def test_decay_reduces_confidence(self):
        state = SmoothingState()
        for _ in range(10):
            state.update(0.85, 0.35, 0.50)
        before = state.smoothed_confidence
        state.decay(0.35)
        assert state.smoothed_confidence < before

    def test_event_start_timestamp_set_on_first_threshold_crossing(self):
        state = SmoothingState()
        assert state.event_start_timestamp is None
        state.update(0.90, 0.35, 0.50)
        assert state.event_start_timestamp is not None

    def test_event_start_timestamp_cleared_on_reset(self):
        state = SmoothingState()
        for _ in range(5):
            state.update(0.90, 0.35, 0.50)
        assert state.event_start_timestamp is not None

        # Feed very low values repeatedly
        for _ in range(20):
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
