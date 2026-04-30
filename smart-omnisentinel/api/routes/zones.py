"""api/routes/zones.py — Zone and threshold profile management."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from api.dependencies import CurrentUser, DBSession, require_role
from api.schemas.zone import (
    ThresholdProfileCreate,
    ThresholdProfileResponse,
    ZoneCreate,
    ZoneResponse,
    ZoneUpdate,
)
from core.constants import UserRole
from core.logger import get_logger
from db.models.threshold_profile import ThresholdProfile
from db.models.zone_config import ZoneConfig

router = APIRouter(tags=["zones"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Zone Config
# ---------------------------------------------------------------------------

zones_router = APIRouter(prefix="/zones")


@zones_router.get("", response_model=List[ZoneResponse])
async def list_zones(db: DBSession, current_user: CurrentUser):
    result = await db.execute(select(ZoneConfig).order_by(ZoneConfig.name))
    return [ZoneResponse.model_validate(z) for z in result.scalars().all()]


@zones_router.post("", response_model=ZoneResponse, status_code=201)
async def create_zone(
    body: ZoneCreate,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    zone = ZoneConfig(**body.model_dump())
    db.add(zone)
    await db.flush()
    return ZoneResponse.model_validate(zone)


@zones_router.put("/{zone_id}", response_model=ZoneResponse)
async def update_zone(
    zone_id: uuid.UUID,
    body: ZoneUpdate,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    zone = await db.get(ZoneConfig, zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found.")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(zone, k, v)
    await db.flush()
    return ZoneResponse.model_validate(zone)


@zones_router.delete("/{zone_id}", status_code=204)
async def delete_zone(
    zone_id: uuid.UUID,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    zone = await db.get(ZoneConfig, zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found.")
    await db.delete(zone)


# ---------------------------------------------------------------------------
# Threshold Profiles
# ---------------------------------------------------------------------------

profiles_router = APIRouter(prefix="/config/thresholds")


@profiles_router.get("", response_model=List[ThresholdProfileResponse])
async def list_profiles(db: DBSession, current_user: CurrentUser):
    result = await db.execute(
        select(ThresholdProfile).order_by(ThresholdProfile.name)
    )
    return [ThresholdProfileResponse.model_validate(p) for p in result.scalars().all()]


@profiles_router.post("", response_model=ThresholdProfileResponse, status_code=201)
async def create_profile(
    body: ThresholdProfileCreate,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    # Only one default profile allowed
    if body.is_default:
        existing = await db.execute(
            select(ThresholdProfile).where(ThresholdProfile.is_default == True)
        )
        current_default = existing.scalar_one_or_none()
        if current_default:
            current_default.is_default = False

    profile = ThresholdProfile(**body.model_dump())
    db.add(profile)
    await db.flush()
    return ThresholdProfileResponse.model_validate(profile)


@profiles_router.put("/{profile_id}", response_model=ThresholdProfileResponse)
async def update_profile(
    profile_id: uuid.UUID,
    body: ThresholdProfileCreate,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    profile = await db.get(ThresholdProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Threshold profile not found.")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(profile, k, v)
    await db.flush()
    return ThresholdProfileResponse.model_validate(profile)
