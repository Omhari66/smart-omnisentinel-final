"""
services/health_monitor/health_reporter.py
--------------------------------------------
Aggregates camera health + service health into one report.
Used by the /api/v1/health endpoint and WebSocket SYSTEM_HEALTH_DEGRADED events.
"""
from __future__ import annotations

import asyncio

from core.constants import WSEventType
from core.event_bus import get_event_bus
from core.logger import get_logger

logger = get_logger(__name__)


class HealthReporter:
    """Subscribes to system_health and pushes WebSocket events on degradation."""

    def __init__(self):
        self._bus = get_event_bus()

    async def run(self) -> None:
        logger.info("health_reporter_started")
        await self._bus.consume("system_health", self._handle_health)

    async def _handle_health(self, payload: dict) -> None:
        status = payload.get("status", "ok")
        if status in ("degraded", "critical"):
            await self._push_ws_degraded(payload, status)

    async def _push_ws_degraded(self, payload: dict, status: str) -> None:
        try:
            from api.schemas.ws_events import SystemHealthDegradedPayload, make_event
            from api.websocket_manager import get_ws_manager

            ws = get_ws_manager()
            p = SystemHealthDegradedPayload(
                component="STORAGE",
                metric="disk_usage_pct",
                value=payload.get("disk_usage_pct", 0),
                threshold=85,
                severity="CRITICAL" if status == "critical" else "WARNING",
            )
            event = make_event(WSEventType.SYSTEM_HEALTH_DEGRADED, p)
            await ws.broadcast(event.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("health_ws_push_failed", error=str(exc))
