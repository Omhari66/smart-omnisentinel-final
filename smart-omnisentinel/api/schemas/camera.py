"""api/schemas/camera.py — Camera CRUD schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from api.schemas.common import OrmBase
from core.constants import CameraStatus


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    stream_url: str = Field(
        min_length=1,
        description="RTSP URL, HTTP stream URL, or local device path",
    )
    location: str = Field(min_length=1, max_length=200)
    zone_id: Optional[uuid.UUID] = None
    threshold_profile_id: Optional[uuid.UUID] = None
    fps_target: int = Field(default=4, ge=1, le=30)
    notes: Optional[str] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    stream_url: Optional[str] = None
    location: Optional[str] = Field(default=None, min_length=1, max_length=200)
    zone_id: Optional[uuid.UUID] = None
    threshold_profile_id: Optional[uuid.UUID] = None
    fps_target: Optional[int] = Field(default=None, ge=1, le=30)
    notes: Optional[str] = None


class ZoneSummary(OrmBase):
    id: uuid.UUID
    name: str
    sensitivity: float


class ThresholdSummary(OrmBase):
    id: uuid.UUID
    name: str


class CameraResponse(OrmBase):
    id: uuid.UUID
    name: str
    stream_url: str
    location: str
    status: CameraStatus
    fps_target: int
    last_seen_at: Optional[datetime]
    notes: Optional[str]
    zone: Optional[ZoneSummary]
    threshold_profile: Optional[ThresholdSummary]
    created_at: datetime
    updated_at: datetime


class CameraHealthResponse(BaseModel):
    camera_id: uuid.UUID
    camera_name: str
    status: CameraStatus
    last_seen_at: Optional[datetime]
    avg_fps: Optional[float]
    missed_frames: Optional[int]
    stream_url_reachable: bool
