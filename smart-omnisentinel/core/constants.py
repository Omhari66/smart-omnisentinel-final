"""
core/constants.py
-----------------
Central definitions for all enumerations and shared constants.
Import from here — never re-define enums in individual modules.
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Event / Incident types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    VIOLENT_INTERACTION = "VIOLENT_INTERACTION"
    FALL_COLLAPSE = "FALL_COLLAPSE"
    CROWD_AGGRESSION = "CROWD_AGGRESSION"
    PERSON_DISTRESS = "PERSON_DISTRESS"


# ---------------------------------------------------------------------------
# Severity tiers (risk-score driven)
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Incident lifecycle status
# ---------------------------------------------------------------------------

class IncidentStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"    # LOW confidence — queued, no alert
    ACTIVE_REVIEW = "ACTIVE_REVIEW"     # MEDIUM — guard notified, awaiting action
    EMERGENCY = "EMERGENCY"             # HIGH — full emergency workflow active
    RESOLVED = "RESOLVED"               # Closed by reviewer or timeout
    FALSE_POSITIVE = "FALSE_POSITIVE"   # Reviewer dismissed as FP


# ---------------------------------------------------------------------------
# Camera health status
# ---------------------------------------------------------------------------

class CameraStatus(str, Enum):
    ACTIVE = "ACTIVE"
    OFFLINE = "OFFLINE"
    DISABLED = "DISABLED"
    RECONNECTING = "RECONNECTING"


# ---------------------------------------------------------------------------
# Reviewer actions
# ---------------------------------------------------------------------------

class ReviewAction(str, Enum):
    CONFIRM = "CONFIRM"             # Reviewer confirms real incident
    DISMISS = "DISMISS"             # Reviewer dismisses as non-incident
    ESCALATE = "ESCALATE"           # Reviewer manually escalates to authorities
    FALSE_POSITIVE = "FALSE_POSITIVE"  # Reviewer flags for model retraining


# ---------------------------------------------------------------------------
# User roles (RBAC)
# ---------------------------------------------------------------------------

class UserRole(str, Enum):
    ADMIN = "ADMIN"             # Full access
    SUPERVISOR = "SUPERVISOR"   # Review + evidence export
    GUARD = "GUARD"             # View alerts + confirm/dismiss
    VIEWER = "VIEWER"           # Read-only dashboard


# ---------------------------------------------------------------------------
# Evidence retention tags
# ---------------------------------------------------------------------------

class RetentionTag(str, Enum):
    TEMP_72H = "TEMP_72H"           # LOW severity — auto-delete after 72h
    STANDARD_30D = "STANDARD_30D"  # MEDIUM severity
    LONG_365D = "LONG_365D"        # HIGH severity
    PERMANENT = "PERMANENT"        # Manually locked by admin


# ---------------------------------------------------------------------------
# WebSocket event types (pushed to dashboard)
# ---------------------------------------------------------------------------

class WSEventType(str, Enum):
    CONNECTED = "CONNECTED"
    INCIDENT_CREATED = "INCIDENT_CREATED"
    ALERT_SEVERITY_CHANGED = "ALERT_SEVERITY_CHANGED"
    CAMERA_STATUS_CHANGED = "CAMERA_STATUS_CHANGED"
    EVIDENCE_READY = "EVIDENCE_READY"
    REVIEW_REQUESTED = "REVIEW_REQUESTED"
    INCIDENT_RESOLVED = "INCIDENT_RESOLVED"
    SYSTEM_HEALTH_DEGRADED = "SYSTEM_HEALTH_DEGRADED"
    PING = "PING"
    PONG = "PONG"


# ---------------------------------------------------------------------------
# Internal event bus channel names
# ---------------------------------------------------------------------------

class Channel(str, Enum):
    FRAMES = "frames"                     # frames.{camera_id}
    EVENT_CANDIDATES = "event_candidates" # event_candidates.{camera_id}
    CONFIRMED_EVENTS = "confirmed_events"
    SCORED_INCIDENTS = "scored_incidents"
    ALERT_ACTIONS = "alert_actions"
    REVIEW_DECISIONS = "review_decisions"
    CAMERA_HEALTH = "camera_health"
    SYSTEM_HEALTH = "system_health"


# ---------------------------------------------------------------------------
# Risk score tier boundaries (defaults; overridden by threshold profiles)
# ---------------------------------------------------------------------------

RISK_LOW_MAX: int = 29
RISK_MEDIUM_MAX: int = 64
RISK_HIGH_MIN: int = 65

# Pre-event buffer: seconds of frames kept in circular buffer
PRE_EVENT_BUFFER_SECONDS: int = 3

# Post-event capture: seconds after event end to keep recording
POST_EVENT_BUFFER_SECONDS: int = 2

# Default escalation timeout for unreviewed MEDIUM incidents (seconds)
DEFAULT_ESCALATION_TIMEOUT: int = 300

# Maximum tracks processed per frame to limit compute
MAX_TRACKS_PER_FRAME: int = 20
