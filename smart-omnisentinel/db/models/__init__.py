"""
db/models/__init__.py
---------------------
Import all ORM models here so that:
1. Alembic's autogenerate can discover all tables.
2. SQLAlchemy's relationship resolution works at startup.
"""

from db.models.alert import Alert
from db.models.audit_log import AuditLog
from db.models.camera import Camera
from db.models.evidence_clip import EvidenceClip
from db.models.incident import Incident
from db.models.model_version import ModelVersion
from db.models.review_action import ReviewActionRecord
from db.models.threshold_profile import ThresholdProfile
from db.models.user import User
from db.models.zone_config import ZoneConfig

__all__ = [
    "Alert", "AuditLog", "Camera", "EvidenceClip", "Incident",
    "ModelVersion", "ReviewActionRecord", "ThresholdProfile", "User", "ZoneConfig",
]
