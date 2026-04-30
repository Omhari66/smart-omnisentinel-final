"""
tests/unit/test_deduplicator.py
---------------------------------
Unit tests for event deduplication and cooldown logic in the EventAnalyzer.
"""
import pytest
import time

from services.event_analysis.event_analyzer import CooldownState, EventAnalyzer


class TestCooldownState:

    def test_cooldown_not_active_initially(self):
        state = CooldownState()
        assert not state.is_active()

    def test_cooldown_active_after_set(self):
        state = CooldownState()
        state.set_cooldown(300.0)
        assert state.is_active()

    def test_cooldown_expires(self):
        state = CooldownState()
        state.set_cooldown(0.05)  # 50ms
        time.sleep(0.1)
        assert not state.is_active()

    def test_cooldown_can_be_reset_by_setting_new(self):
        state = CooldownState()
        state.set_cooldown(0.05)
        time.sleep(0.1)
        assert not state.is_active()
        # Set again — should be active
        state.set_cooldown(300.0)
        assert state.is_active()


class TestEventAnalyzerDedup:

    @pytest.mark.asyncio
    async def test_zone_suppression_blocks_event(self):
        """Events suppressed for a zone must not be emitted."""
        analyzer = EventAnalyzer(
            suppressed_event_types={"cam-001": {"VIOLENT_INTERACTION"}}
        )
        published = []

        # Patch the bus to capture published events
        original_publish = analyzer._bus.publish
        async def capture(channel, msg):
            published.append(msg)
        analyzer._bus.publish = capture

        await analyzer.handle_candidate({
            "camera_id": "cam-001",
            "event_type": "VIOLENT_INTERACTION",
            "raw_confidence": 0.95,
            "people_count": 2,
            "track_ids": [1, 2],
            "motion_magnitude": 5.0,
            "timestamp": time.time(),
        })

        assert len(published) == 0, "Suppressed event should not be published"

    @pytest.mark.asyncio
    async def test_cooldown_blocks_event(self):
        """Events within cooldown window must not be confirmed."""
        analyzer = EventAnalyzer(persistence_frames=1, ema_alpha=1.0)
        published = []

        async def capture(channel, msg):
            published.append(msg)
        analyzer._bus.publish = capture

        # Set active cooldown
        analyzer.set_cooldown("cam-001", "VIOLENT_INTERACTION")

        await analyzer.handle_candidate({
            "camera_id": "cam-001",
            "event_type": "VIOLENT_INTERACTION",
            "raw_confidence": 0.99,
            "people_count": 2,
            "track_ids": [1],
            "motion_magnitude": 8.0,
            "timestamp": time.time(),
        })

        assert len(published) == 0, "Cooldown should block event emission"

    @pytest.mark.asyncio
    async def test_persistence_gates_emission(self):
        """Events must persist for required frames before being emitted."""
        required = 5
        analyzer = EventAnalyzer(
            persistence_frames=required,
            ema_alpha=1.0,  # No smoothing — raw confidence used directly
            confidence_threshold=0.5,
        )
        published = []
        async def capture(channel, msg):
            published.append(msg)
        analyzer._bus.publish = capture

        camera_id = "cam-persist-test"
        candidate = {
            "camera_id": camera_id,
            "event_type": "FALL_COLLAPSE",
            "raw_confidence": 0.80,
            "people_count": 1,
            "track_ids": [5],
            "motion_magnitude": 2.0,
            "timestamp": time.time(),
        }

        # Feed required - 1 frames — should NOT emit
        for _ in range(required - 1):
            await analyzer.handle_candidate(candidate)
        assert len(published) == 0, f"Should not emit before {required} frames"

        # Feed one more — should NOW emit
        await analyzer.handle_candidate(candidate)
        assert len(published) >= 1, "Should emit after persistence threshold"
