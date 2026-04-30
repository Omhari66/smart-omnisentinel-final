"""
api/routes/ws.py
----------------
WebSocket endpoint for real-time dashboard events.
Authentication via JWT passed as query parameter (standard for WS).
On connect: sends CONNECTED snapshot with current system state.
Keeps connection alive with periodic PING/PONG.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import func, select

from api.schemas.ws_events import ConnectedPayload, WSEventEnvelope, WSEventType
from api.websocket_manager import get_ws_manager
from core.constants import CameraStatus, IncidentStatus
from core.logger import get_logger
from core.security import decode_token
from core.exceptions import AuthenticationError

router = APIRouter(tags=["websocket"])
logger = get_logger(__name__)

PING_INTERVAL_SECONDS = 30


@router.websocket("/ws/events")
async def websocket_events(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
):
    """
    WebSocket endpoint for live event streaming.

    Connect with: ws://host/api/v1/ws/events?token=<access_token>

    Client should respond to PING events with PONG to maintain connection.
    Server disconnects clients that fail to PONG within 10 seconds.
    """
    # Authenticate before accepting connection
    try:
        payload = decode_token(token)
        user_id = payload.get("sub", "unknown")
        role = payload.get("role", "VIEWER")
    except AuthenticationError as exc:
        await websocket.close(code=4001, reason=str(exc))
        return

    connection_id = str(uuid.uuid4())
    ws_manager = get_ws_manager()

    await ws_manager.connect(
        connection_id=connection_id,
        websocket=websocket,
        user_id=user_id,
        role=role,
    )

    # Send initial connection snapshot
    try:
        await _send_connected_snapshot(websocket)
    except Exception as exc:
        logger.warning("ws_snapshot_failed", error=str(exc))

    # Start ping task
    ping_task = asyncio.create_task(
        _ping_loop(websocket, connection_id)
    )

    try:
        while True:
            try:
                # Wait for client messages (PONG or subscription requests)
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=PING_INTERVAL_SECONDS + 10
                )
                message = json.loads(raw)
                event_type = message.get("event_type")

                if event_type == "PONG":
                    pass  # Client is alive, nothing to do
                else:
                    logger.debug(
                        "ws_unknown_message",
                        connection_id=connection_id,
                        event_type=event_type,
                    )

            except asyncio.TimeoutError:
                # Client missed PONG, disconnect
                logger.warning(
                    "ws_client_timeout", connection_id=connection_id
                )
                break

    except WebSocketDisconnect:
        logger.info("ws_client_disconnected_gracefully", connection_id=connection_id)
    except Exception as exc:
        logger.error("ws_error", connection_id=connection_id, error=str(exc))
    finally:
        ping_task.cancel()
        await ws_manager.disconnect(connection_id)


async def _send_connected_snapshot(websocket: WebSocket) -> None:
    """
    Send CONNECTED event with current system state snapshot.
    Does a lightweight DB query for active counts.
    """
    # Import here to avoid circular imports at module level
    from db.session import get_session_factory

    factory = get_session_factory()
    async with factory() as db:
        # Active incidents
        active_inc = await db.execute(
            select(func.count()).where(
                func.cast(IncidentStatus.EMERGENCY, type_=None) is not None
            )
        )
        # Simpler approach: count by status
        pending_result = await db.execute(
            select(func.count(IncidentStatus.PENDING_REVIEW))
        )

        from db.models.camera import Camera
        from db.models.incident import Incident

        cameras_online = (await db.execute(
            select(func.count()).select_from(Camera)
            .where(Camera.status == CameraStatus.ACTIVE)
        )).scalar_one()

        cameras_offline = (await db.execute(
            select(func.count()).select_from(Camera)
            .where(Camera.status == CameraStatus.OFFLINE)
        )).scalar_one()

        active_incidents = (await db.execute(
            select(func.count()).select_from(Incident)
            .where(Incident.status.in_([
                IncidentStatus.ACTIVE_REVIEW,
                IncidentStatus.EMERGENCY,
            ]))
        )).scalar_one()

        pending_reviews = (await db.execute(
            select(func.count()).select_from(Incident)
            .where(Incident.status == IncidentStatus.PENDING_REVIEW)
        )).scalar_one()

    payload = ConnectedPayload(
        active_incidents=active_incidents,
        cameras_online=cameras_online,
        cameras_offline=cameras_offline,
        pending_reviews=pending_reviews,
        server_time=datetime.now(tz=timezone.utc),
    )
    envelope = WSEventEnvelope(
        event_type=WSEventType.CONNECTED,
        payload=payload.model_dump(mode="json"),
    )
    await websocket.send_text(envelope.model_dump_json())


async def _ping_loop(websocket: WebSocket, connection_id: str) -> None:
    """Send PING every 30 seconds to keep the connection alive."""
    while True:
        await asyncio.sleep(PING_INTERVAL_SECONDS)
        try:
            ping = WSEventEnvelope(
                event_type=WSEventType.PING,
                payload={"ts": datetime.now(tz=timezone.utc).isoformat()},
            )
            await websocket.send_text(ping.model_dump_json())
        except Exception:
            break
