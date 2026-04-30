"""
services/risk_engine/risk_scorer.py
-------------------------------------
Multi-signal risk scoring engine.
Converts a smoothed event candidate into a 0-100 risk score
and assigns a severity tier (LOW / MEDIUM / HIGH).

Formula:
  RISK = clamp(
      W_conf   * confidence_score         # 30 pts
    + W_dur    * duration_factor          # 15 pts
    + W_pop    * people_count_factor      # 10 pts
    + W_motion * motion_intensity         # 10 pts
    + W_repeat * recent_alert_factor      # 10 pts
    + W_zone   * zone_sensitivity         # 15 pts
    + W_time   * time_of_day_factor       #  5 pts
    + W_human  * human_confirmation_bonus #  5 pts
  , 0, 100)

Each factor is 0.0–1.0; weights sum to 100.
Thresholds (LOW_MAX, MEDIUM_MAX) are pulled from the camera's threshold profile.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.constants import RISK_HIGH_MIN, RISK_LOW_MAX, RISK_MEDIUM_MAX, Severity
from core.logger import get_logger

logger = get_logger(__name__)

# Weight table (must sum to 100)
W_CONF = 30
W_DUR = 15
W_POP = 10
W_MOTION = 10
W_REPEAT = 10
W_ZONE = 15
W_TIME = 5
W_HUMAN = 5

# Duration at which duration_factor reaches 1.0 (seconds)
DURATION_SATURATION_S = 30.0

# People count at which people_count_factor reaches 1.0
PEOPLE_COUNT_SATURATION = 5

# Motion magnitude at which motion_intensity reaches 1.0
MOTION_SATURATION = 8.0

# Hours considered "late night / early morning" for increased risk
HIGH_RISK_HOURS = set(range(22, 24)) | set(range(0, 6))


@dataclass
class RiskSignals:
    """All contextual signals collected before scoring."""
    confidence: float          # Smoothed classifier confidence 0.0–1.0
    duration_seconds: float    # How long the event has persisted
    people_count: int          # Number of persons involved
    motion_magnitude: float    # Scene optical flow magnitude
    recent_alert: bool         # Same camera had alert in last N minutes
    zone_sensitivity: float    # 0.0–1.0 from zone config
    hour_of_day: int           # 0–23
    human_confirmed: bool      # Reviewer has confirmed this event


@dataclass
class RiskScore:
    """Output of the risk scoring engine."""
    score: int                  # 0–100
    severity: Severity
    factor_breakdown: Dict[str, float]


def compute_risk_score(
    signals: RiskSignals,
    risk_low_max: int = RISK_LOW_MAX,
    risk_medium_max: int = RISK_MEDIUM_MAX,
) -> RiskScore:
    """
    Compute risk score from contextual signals.
    Returns RiskScore with score, severity tier, and per-factor breakdown.
    """
    # --- Compute each factor (0.0–1.0) ---

    conf_factor = float(signals.confidence)

    dur_factor = min(signals.duration_seconds / DURATION_SATURATION_S, 1.0)

    pop_factor = min(signals.people_count / PEOPLE_COUNT_SATURATION, 1.0)

    motion_factor = min(signals.motion_magnitude / MOTION_SATURATION, 1.0)

    repeat_factor = 1.0 if signals.recent_alert else 0.0

    zone_factor = float(signals.zone_sensitivity)

    time_factor = 1.0 if signals.hour_of_day in HIGH_RISK_HOURS else 0.3

    human_factor = 1.0 if signals.human_confirmed else 0.0

    # --- Weighted sum ---
    raw_score = (
        W_CONF   * conf_factor
        + W_DUR    * dur_factor
        + W_POP    * pop_factor
        + W_MOTION * motion_factor
        + W_REPEAT * repeat_factor
        + W_ZONE   * zone_factor
        + W_TIME   * time_factor
        + W_HUMAN  * human_factor
    )

    score = int(max(0, min(100, round(raw_score))))

    # --- Tier assignment ---
    if score <= risk_low_max:
        severity = Severity.LOW
    elif score <= risk_medium_max:
        severity = Severity.MEDIUM
    else:
        severity = Severity.HIGH

    breakdown = {
        "confidence": round(conf_factor, 3),
        "duration": round(dur_factor, 3),
        "people_count": round(pop_factor, 3),
        "motion": round(motion_factor, 3),
        "recent_alert": round(repeat_factor, 3),
        "zone_sensitivity": round(zone_factor, 3),
        "time_of_day": round(time_factor, 3),
        "human_confirmed": round(human_factor, 3),
        "raw_score": round(raw_score, 2),
    }

    logger.debug(
        "risk_score_computed",
        score=score,
        severity=severity,
        breakdown=breakdown,
    )

    return RiskScore(score=score, severity=severity, factor_breakdown=breakdown)


def build_signals_from_event(
    event: Dict[str, Any],
    zone_sensitivity: float = 0.7,
    recent_alert: bool = False,
    human_confirmed: bool = False,
    event_start_timestamp: Optional[float] = None,
) -> RiskSignals:
    """
    Convenience builder: constructs RiskSignals from the event_analysis
    output dict (as published on the 'confirmed_events' channel).
    """
    now = time.time()
    duration = now - event_start_timestamp if event_start_timestamp else 0.0
    hour = datetime.now(tz=timezone.utc).hour

    return RiskSignals(
        confidence=float(event.get("smoothed_confidence", 0.0)),
        duration_seconds=duration,
        people_count=int(event.get("people_count", 1)),
        motion_magnitude=float(event.get("motion_magnitude", 0.0)),
        recent_alert=recent_alert,
        zone_sensitivity=zone_sensitivity,
        hour_of_day=hour,
        human_confirmed=human_confirmed,
    )
