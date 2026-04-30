"""api/schemas/zone.py — Zone and threshold profile schemas."""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from api.schemas.common import OrmBase


class ZoneCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = None
    sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
    suppress_event_types: Optional[List[str]] = None
    peak_hours_schedule: Optional[List[Any]] = None


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    sensitivity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    suppress_event_types: Optional[List[str]] = None
    peak_hours_schedule: Optional[List[Any]] = None


class ZoneResponse(OrmBase):
    id: uuid.UUID
    name: str
    description: Optional[str]
    sensitivity: float
    suppress_event_types: Optional[List[str]]
    peak_hours_schedule: Optional[List[Any]]


class ThresholdProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    violence_confidence_min: float = Field(default=0.72, ge=0.0, le=1.0)
    fall_confidence_min: float = Field(default=0.68, ge=0.0, le=1.0)
    crowd_confidence_min: float = Field(default=0.65, ge=0.0, le=1.0)
    persistence_frames: int = Field(default=12, ge=1, le=60)
    risk_low_max: int = Field(default=29, ge=0, le=99)
    risk_medium_max: int = Field(default=64, ge=1, le=99)
    ema_alpha: float = Field(default=0.35, ge=0.01, le=1.0)
    cooldown_seconds: int = Field(default=300, ge=30, le=3600)
    escalation_timeout_seconds: int = Field(default=300, ge=60, le=3600)
    is_default: bool = False


class ThresholdProfileResponse(OrmBase):
    id: uuid.UUID
    name: str
    violence_confidence_min: float
    fall_confidence_min: float
    crowd_confidence_min: float
    persistence_frames: int
    risk_low_max: int
    risk_medium_max: int
    ema_alpha: float
    cooldown_seconds: int
    escalation_timeout_seconds: int
    is_default: bool
