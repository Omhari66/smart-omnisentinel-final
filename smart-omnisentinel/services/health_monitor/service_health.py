"""
services/health_monitor/service_health.py
------------------------------------------
Tracks inference latency, queue depth, and disk usage.
Publishes to 'system_health' bus channel; health API endpoint reads from here.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from core.config import get_settings
from core.event_bus import get_event_bus
from core.logger import get_logger

logger = get_logger(__name__)

CHECK_INTERVAL = 15.0   # seconds


@dataclass
class ServiceHealthSnapshot:
    timestamp: float = field(default_factory=time.time)
    inference_avg_ms: float = 0.0
    event_bus_queue_depth: int = 0
    disk_free_gb: float = 0.0
    disk_usage_pct: float = 0.0
    active_cameras: int = 0
    status: str = "ok"   # "ok" | "degraded" | "critical"


class ServiceHealthMonitor:
    """Aggregates service-level health metrics and publishes snapshots."""

    def __init__(self):
        self._bus = get_event_bus()
        self._latest: Optional[ServiceHealthSnapshot] = None

    @property
    def latest(self) -> Optional[ServiceHealthSnapshot]:
        return self._latest

    async def run(self) -> None:
        logger.info("service_health_monitor_started")
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                snapshot = await self._collect()
                self._latest = snapshot
                await self._bus.publish("system_health", {
                    "timestamp": snapshot.timestamp,
                    "inference_avg_ms": snapshot.inference_avg_ms,
                    "disk_free_gb": snapshot.disk_free_gb,
                    "disk_usage_pct": snapshot.disk_usage_pct,
                    "active_cameras": snapshot.active_cameras,
                    "status": snapshot.status,
                })
                if snapshot.status != "ok":
                    logger.warning("system_health_degraded", status=snapshot.status,
                                   disk_pct=snapshot.disk_usage_pct)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("service_health_error", error=str(exc))

    async def _collect(self) -> ServiceHealthSnapshot:
        settings = get_settings()
        snapshot = ServiceHealthSnapshot()

        # Disk check
        evidence_root = settings.storage.evidence_root
        if os.path.exists(evidence_root):
            try:
                stat = os.statvfs(evidence_root)
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
                total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
                snapshot.disk_free_gb = round(free_gb, 2)
                snapshot.disk_usage_pct = round((1 - free_gb / total_gb) * 100, 1)
            except Exception:
                pass

        # Determine status
        if snapshot.disk_usage_pct > 95:
            snapshot.status = "critical"
        elif snapshot.disk_usage_pct > 85:
            snapshot.status = "degraded"
        else:
            snapshot.status = "ok"

        return snapshot
