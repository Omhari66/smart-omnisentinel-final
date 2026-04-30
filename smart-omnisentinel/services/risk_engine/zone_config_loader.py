"""
services/risk_engine/zone_config_loader.py
-------------------------------------------
Loads and caches zone configurations for the risk engine.
Caches results for 60 seconds to avoid repeated DB queries per frame.
"""
from __future__ import annotations

import time
from typing import Optional
from uuid import UUID

from core.logger import get_logger

logger = get_logger(__name__)

CACHE_TTL = 60.0  # seconds


class ZoneConfigCache:
    """Simple TTL-based in-memory cache for zone configs."""

    def __init__(self, ttl: float = CACHE_TTL):
        self._ttl = ttl
        self._cache: dict = {}   # camera_id → (zone_data, expires_at)

    async def get_zone_for_camera(self, camera_id: str) -> Optional[dict]:
        """
        Returns zone config dict for the camera, or None if no zone assigned.
        Uses cache to avoid per-frame DB hits.
        """
        now = time.time()
        if camera_id in self._cache:
            data, expires = self._cache[camera_id]
            if now < expires:
                return data

        # Cache miss or expired — fetch from DB
        zone_data = await self._fetch_zone(camera_id)
        self._cache[camera_id] = (zone_data, now + self._ttl)
        return zone_data

    async def _fetch_zone(self, camera_id: str) -> Optional[dict]:
        try:
            from db.session import get_session_factory
            from db.models.camera import Camera
            from sqlalchemy import select

            factory = get_session_factory()
            async with factory() as db:
                result = await db.execute(
                    select(Camera).where(Camera.id == UUID(camera_id))
                )
                camera = result.scalar_one_or_none()
                if camera and camera.zone:
                    return {
                        "name": camera.zone.name,
                        "sensitivity": camera.zone.sensitivity,
                        "suppress_event_types": camera.zone.suppress_event_types or [],
                        "peak_hours_schedule": camera.zone.peak_hours_schedule or [],
                    }
        except Exception as exc:
            logger.warning("zone_fetch_failed", camera_id=camera_id, error=str(exc))
        return None

    def invalidate(self, camera_id: str) -> None:
        self._cache.pop(camera_id, None)


# Singleton
_cache: Optional[ZoneConfigCache] = None


def get_zone_cache() -> ZoneConfigCache:
    global _cache
    if _cache is None:
        _cache = ZoneConfigCache()
    return _cache
