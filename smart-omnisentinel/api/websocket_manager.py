"""
api/websocket_manager.py
------------------------
Manages the pool of active WebSocket connections.
Handles broadcast, targeted push, and graceful disconnect.
All services call ws_manager.broadcast() to push events to the dashboard.
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional, Set
from uuid import UUID

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from core.logger import get_logger

logger = get_logger(__name__)


class ConnectionMeta:
    """Metadata stored per connected WebSocket session."""

    __slots__ = ("websocket", "user_id", "role", "subscriptions")

    def __init__(
        self,
        websocket: WebSocket,
        user_id: str,
        role: str,
        subscriptions: Optional[Set[str]] = None,
    ):
        self.websocket = websocket
        self.user_id = user_id
        self.role = role
        # Empty set = subscribe to all event types
        self.subscriptions: Set[str] = subscriptions or set()


class WebSocketManager:
    """
    Thread-safe (asyncio-safe) WebSocket connection pool.
    Supports broadcast to all clients and targeted push to specific users.
    """

    def __init__(self) -> None:
        # connection_id (str) → ConnectionMeta
        self._connections: Dict[str, ConnectionMeta] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        connection_id: str,
        websocket: WebSocket,
        user_id: str,
        role: str,
    ) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[connection_id] = ConnectionMeta(
                websocket=websocket, user_id=user_id, role=role
            )
        logger.info(
            "ws_client_connected",
            connection_id=connection_id,
            user_id=user_id,
            role=role,
            total_connections=len(self._connections),
        )

    async def disconnect(self, connection_id: str) -> None:
        async with self._lock:
            self._connections.pop(connection_id, None)
        logger.info(
            "ws_client_disconnected",
            connection_id=connection_id,
            total_connections=len(self._connections),
        )

    async def broadcast(self, payload: dict) -> None:
        """Send payload to all connected clients."""
        if not self._connections:
            return

        message = json.dumps(payload, default=str)
        dead_connections: list[str] = []

        for conn_id, meta in list(self._connections.items()):
            if meta.websocket.client_state != WebSocketState.CONNECTED:
                dead_connections.append(conn_id)
                continue
            try:
                await meta.websocket.send_text(message)
            except Exception as exc:
                logger.warning(
                    "ws_send_failed",
                    connection_id=conn_id,
                    error=str(exc),
                )
                dead_connections.append(conn_id)

        # Clean up dead connections
        for conn_id in dead_connections:
            await self.disconnect(conn_id)

    async def send_to_user(self, user_id: str, payload: dict) -> None:
        """Send payload to all connections belonging to a specific user."""
        message = json.dumps(payload, default=str)
        for conn_id, meta in list(self._connections.items()):
            if meta.user_id != user_id:
                continue
            try:
                await meta.websocket.send_text(message)
            except Exception:
                await self.disconnect(conn_id)

    @property
    def active_count(self) -> int:
        return len(self._connections)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_ws_manager: Optional[WebSocketManager] = None


def get_ws_manager() -> WebSocketManager:
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
