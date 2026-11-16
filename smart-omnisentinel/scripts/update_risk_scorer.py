import os

path = r"C:\Users\OMHARI\Development\Projects\ML_Projects\smart-omnisentinel-final\smart-omnisentinel\services\risk_engine\risk_scorer.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update RiskSignals
old_signals = """@dataclass
class RiskSignals:
    \"\"\"All contextual signals collected before scoring.\"\"\"
    confidence: float          # Smoothed classifier confidence 0.0–1.0
    duration_seconds: float    # How long the event has persisted
    people_count: int          # Number of persons involved
    motion_magnitude: float    # Scene optical flow magnitude
    recent_alert: bool         # Same camera had alert in last N minutes
    zone_sensitivity: float    # 0.0–1.0 from zone config
    hour_of_day: int           # 0–23
    human_confirmed: bool      # Reviewer has confirmed this event"""

new_signals = """@dataclass
class RiskSignals:
    \"\"\"All contextual signals collected before scoring.\"\"\"
    confidence: float          # Smoothed classifier confidence 0.0–1.0
    duration_seconds: float    # How long the event has persisted
    people_count: int          # Number of persons involved
    motion_magnitude: float    # Scene optical flow magnitude
    recent_alert: bool         # Same camera had alert in last N minutes
    zone_sensitivity: float    # 0.0–1.0 from zone config
    hour_of_day: int           # 0–23
    human_confirmed: bool      # Reviewer has confirmed this event
    weapon_detected: bool = False  # Secondary signal: weapon within track proximity
    weapon_type: Optional[str] = None  # knife, baseball_bat, bottle, scissors"""

if old_signals in content:
    content = content.replace(old_signals, new_signals)
    print("1. RiskSignals updated")
else:
    print("1. RiskSignals already updated or not found")

# 2. Update RiskScore
old_score = """@dataclass
class RiskScore:
    \"\"\"Output of the risk scoring engine.\"\"\"
    score: int                  # 0–100
    severity: Severity
    factor_breakdown: Dict[str, float]"""

new_score = """@dataclass
class RiskScore:
    \"\"\"Output of the risk scoring engine.\"\"\"
    score: int                  # 0–100
    severity: Severity
    factor_breakdown: Dict[str, float]
    weapon_detected: bool = False
    weapon_type: Optional[str] = None"""

if old_score in content:
    content = content.replace(old_score, new_score)
    print("2. RiskScore updated")
else:
    print("2. RiskScore already updated or not found")

# 3. Update raw_score calculation with weapon multiplier
old_calc = """    score = int(max(0, min(100, round(raw_score))))"""
new_calc = """    # Secondary signal weapon escalation (Day 12):
    # If weapon detected within 0.5m of an active track -> 1.5x multiplier
    weapon_mult = 1.5 if signals.weapon_detected else 1.0
    raw_score = raw_score * weapon_mult
    score = int(max(0, min(100, round(raw_score))))"""

if old_calc in content:
    content = content.replace(old_calc, new_calc, 1)
    print("3. raw_score calculation updated")
else:
    print("3. raw_score calculation already updated or not found")

# 4. Update breakdown and return
old_breakdown = """    breakdown = {
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

    return RiskScore(score=score, severity=severity, factor_breakdown=breakdown)"""

new_breakdown = """    breakdown = {
        "confidence": round(conf_factor, 3),
        "duration": round(dur_factor, 3),
        "people_count": round(pop_factor, 3),
        "motion": round(motion_factor, 3),
        "recent_alert": round(repeat_factor, 3),
        "zone_sensitivity": round(zone_factor, 3),
        "time_of_day": round(time_factor, 3),
        "human_confirmed": round(human_factor, 3),
        "raw_score": round(raw_score, 2),
        "weapon_multiplier": weapon_mult,
        "weapon_detected": signals.weapon_detected,
        "weapon_type": signals.weapon_type,
    }

    logger.debug(
        "risk_score_computed",
        score=score,
        severity=severity,
        breakdown=breakdown,
        weapon_detected=signals.weapon_detected,
        weapon_type=signals.weapon_type,
    )

    return RiskScore(
        score=score,
        severity=severity,
        factor_breakdown=breakdown,
        weapon_detected=signals.weapon_detected,
        weapon_type=signals.weapon_type,
    )"""

if old_breakdown in content:
    content = content.replace(old_breakdown, new_breakdown, 1)
    print("4. breakdown updated")
else:
    print("4. breakdown already updated or not found")

# 5. Update build_signals_from_event
old_builder = """    return RiskSignals(
        confidence=float(event.get("smoothed_confidence", 0.0)),
        duration_seconds=duration,
        people_count=int(event.get("people_count", 1)),
        motion_magnitude=float(event.get("motion_magnitude", 0.0)),
        recent_alert=recent_alert,
        zone_sensitivity=zone_sensitivity,
        hour_of_day=hour,
        human_confirmed=human_confirmed,
    )"""

new_builder = """    return RiskSignals(
        confidence=float(event.get("smoothed_confidence", 0.0)),
        duration_seconds=duration,
        people_count=int(event.get("people_count", 1)),
        motion_magnitude=float(event.get("motion_magnitude", 0.0)),
        recent_alert=recent_alert,
        zone_sensitivity=zone_sensitivity,
        hour_of_day=hour,
        human_confirmed=human_confirmed,
        weapon_detected=bool(event.get("weapon_detected", False)),
        weapon_type=event.get("weapon_type", None),
    )"""

if old_builder in content:
    content = content.replace(old_builder, new_builder, 1)
    print("5. builder updated")
else:
    print("5. builder already updated or not found")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Finished updating risk_scorer.py")

