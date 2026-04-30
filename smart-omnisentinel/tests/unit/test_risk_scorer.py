"""
tests/unit/test_risk_scorer.py
-------------------------------
Unit tests for the risk scoring engine.
No DB, no external services — pure function tests.
"""

import pytest

from core.constants import Severity
from services.risk_engine.risk_scorer import (
    RiskSignals,
    compute_risk_score,
    build_signals_from_event,
)


def make_signals(**overrides) -> RiskSignals:
    """Build a RiskSignals with sensible defaults, applying overrides."""
    defaults = dict(
        confidence=0.80,
        duration_seconds=15.0,
        people_count=2,
        motion_magnitude=4.0,
        recent_alert=False,
        zone_sensitivity=0.7,
        hour_of_day=14,  # 2 PM — daytime
        human_confirmed=False,
    )
    defaults.update(overrides)
    return RiskSignals(**defaults)


class TestRiskScoreComputation:

    def test_high_confidence_daytime_medium_risk(self):
        signals = make_signals(confidence=0.85, duration_seconds=10.0)
        result = compute_risk_score(signals)
        # Should be MEDIUM: high confidence but short duration, daytime, no repeat
        assert result.score > 0
        assert result.severity in (Severity.MEDIUM, Severity.HIGH)

    def test_low_confidence_is_low_severity(self):
        signals = make_signals(
            confidence=0.30,
            duration_seconds=2.0,
            people_count=1,
            motion_magnitude=0.5,
            zone_sensitivity=0.3,
        )
        result = compute_risk_score(signals)
        assert result.severity == Severity.LOW
        assert result.score <= 29

    def test_high_risk_nighttime_crowd(self):
        signals = make_signals(
            confidence=0.92,
            duration_seconds=45.0,
            people_count=8,
            motion_magnitude=9.0,
            recent_alert=True,
            zone_sensitivity=0.9,
            hour_of_day=2,  # 2 AM
        )
        result = compute_risk_score(signals)
        assert result.severity == Severity.HIGH
        assert result.score >= 65

    def test_human_confirmation_boosts_score(self):
        base = make_signals(confidence=0.65, human_confirmed=False)
        confirmed = make_signals(confidence=0.65, human_confirmed=True)
        base_result = compute_risk_score(base)
        confirmed_result = compute_risk_score(confirmed)
        assert confirmed_result.score > base_result.score

    def test_score_clamped_to_100(self):
        signals = make_signals(
            confidence=1.0,
            duration_seconds=999.0,
            people_count=100,
            motion_magnitude=100.0,
            recent_alert=True,
            zone_sensitivity=1.0,
            hour_of_day=3,
            human_confirmed=True,
        )
        result = compute_risk_score(signals)
        assert result.score <= 100

    def test_score_not_negative(self):
        signals = make_signals(
            confidence=0.0,
            duration_seconds=0.0,
            people_count=0,
            motion_magnitude=0.0,
            recent_alert=False,
            zone_sensitivity=0.0,
            hour_of_day=12,
            human_confirmed=False,
        )
        result = compute_risk_score(signals)
        assert result.score >= 0

    def test_factor_breakdown_present(self):
        signals = make_signals()
        result = compute_risk_score(signals)
        expected_keys = {
            "confidence", "duration", "people_count", "motion",
            "recent_alert", "zone_sensitivity", "time_of_day",
            "human_confirmed", "raw_score"
        }
        assert expected_keys.issubset(result.factor_breakdown.keys())

    def test_recent_alert_increases_score(self):
        no_repeat = make_signals(recent_alert=False)
        with_repeat = make_signals(recent_alert=True)
        assert compute_risk_score(with_repeat).score > compute_risk_score(no_repeat).score

    def test_custom_tier_thresholds(self):
        signals = make_signals(confidence=0.90, duration_seconds=30.0, zone_sensitivity=0.8)
        # Shift medium threshold up so same score becomes LOW
        result_strict = compute_risk_score(signals, risk_low_max=80, risk_medium_max=90)
        result_default = compute_risk_score(signals)
        # Strict thresholds should give LOWER or equal severity
        assert (
            [Severity.LOW, Severity.MEDIUM, Severity.HIGH].index(result_strict.severity)
            <= [Severity.LOW, Severity.MEDIUM, Severity.HIGH].index(result_default.severity)
        )


class TestBuildSignalsFromEvent:

    def test_build_from_event_dict(self):
        import time
        event = {
            "smoothed_confidence": 0.82,
            "people_count": 3,
            "motion_magnitude": 5.0,
        }
        signals = build_signals_from_event(
            event=event,
            zone_sensitivity=0.75,
            recent_alert=True,
            event_start_timestamp=time.time() - 20.0,
        )
        assert signals.confidence == pytest.approx(0.82)
        assert signals.people_count == 3
        assert signals.duration_seconds == pytest.approx(20.0, abs=2.0)
        assert signals.zone_sensitivity == 0.75
        assert signals.recent_alert is True

    def test_build_with_missing_fields(self):
        signals = build_signals_from_event(event={})
        assert signals.confidence == 0.0
        assert signals.people_count == 1
        assert signals.motion_magnitude == 0.0
