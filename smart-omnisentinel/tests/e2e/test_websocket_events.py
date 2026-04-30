"""
tests/e2e/test_websocket_events.py
------------------------------------
Tests for WebSocket event delivery and payload schemas.
Verifies that all WSEventType payloads serialize correctly
and that the WebSocketManager broadcasts to subscribers.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import pytest


class TestWSEventSchemas:
    """Verify all WS event payloads serialize to valid JSON."""

    def test_incident_created_payload_serializes(self):
        from api.schemas.ws_events import IncidentCreatedPayload, make_event
        from core.constants import EventType, Severity, WSEventType

        payload = IncidentCreatedPayload(
            incident_id=uuid.uuid4(),
            camera_id=uuid.uuid4(),
            camera_name="Test Camera",
            location="Building A",
            event_type=EventType.VIOLENT_INTERACTION,
            severity=Severity.HIGH,
            risk_score=78,
            confidence=0.87,
            detected_at=datetime.now(tz=timezone.utc),
        )
        event = make_event(WSEventType.INCIDENT_CREATED, payload)
        serialized = event.model_dump_json()
        parsed = json.loads(serialized)
        assert parsed["event_type"] == "INCIDENT_CREATED"
        assert "payload" in parsed
        assert parsed["payload"]["risk_score"] == 78

    def test_alert_severity_changed_serializes(self):
        from api.schemas.ws_events import AlertSeverityChangedPayload, make_event
        from core.constants import Severity, WSEventType

        payload = AlertSeverityChangedPayload(
            incident_id=uuid.uuid4(),
            previous_severity=Severity.MEDIUM,
            new_severity=Severity.HIGH,
            reason="AUTO_ESCALATED_TIMEOUT",
            changed_at=datetime.now(tz=timezone.utc),
        )
        event = make_event(WSEventType.ALERT_SEVERITY_CHANGED, payload)
        serialized = json.loads(event.model_dump_json())
        assert serialized["payload"]["reason"] == "AUTO_ESCALATED_TIMEOUT"
        assert serialized["payload"]["new_severity"] == "HIGH"

    def test_camera_status_changed_serializes(self):
        from api.schemas.ws_events import CameraStatusChangedPayload, make_event
        from core.constants import CameraStatus, WSEventType

        payload = CameraStatusChangedPayload(
            camera_id=uuid.uuid4(),
            camera_name="Gate Camera",
            previous_status=CameraStatus.ACTIVE,
            new_status=CameraStatus.OFFLINE,
            reason="STREAM_TIMEOUT",
            changed_at=datetime.now(tz=timezone.utc),
        )
        event = make_event(WSEventType.CAMERA_STATUS_CHANGED, payload)
        parsed = json.loads(event.model_dump_json())
        assert parsed["payload"]["new_status"] == "OFFLINE"

    def test_evidence_ready_serializes(self):
        from api.schemas.ws_events import EvidenceReadyPayload, make_event
        from core.constants import WSEventType

        payload = EvidenceReadyPayload(
            incident_id=uuid.uuid4(),
            evidence_id=uuid.uuid4(),
            duration_seconds=58.4,
        )
        event = make_event(WSEventType.EVIDENCE_READY, payload)
        parsed = json.loads(event.model_dump_json())
        assert parsed["payload"]["duration_seconds"] == pytest.approx(58.4)

    def test_all_event_types_have_unique_ids(self):
        """Each make_event call should produce a unique event_id."""
        from api.schemas.ws_events import EvidenceReadyPayload, make_event
        from core.constants import WSEventType

        payload = EvidenceReadyPayload(
            incident_id=uuid.uuid4(),
            evidence_id=uuid.uuid4(),
            duration_seconds=10.0,
        )
        event_ids = {
            make_event(WSEventType.EVIDENCE_READY, payload).event_id
            for _ in range(10)
        }
        assert len(event_ids) == 10, "Each event must have a unique ID"


class TestWebSocketManager:
    """Test WebSocketManager connection pool behavior."""

    @pytest.mark.asyncio
    async def test_broadcast_to_no_clients_does_not_raise(self):
        """Broadcasting with no connections should be a no-op."""
        from api.websocket_manager import WebSocketManager

        manager = WebSocketManager()
        # Should not raise even with no connections
        await manager.broadcast({"event_type": "TEST", "payload": {}})

    @pytest.mark.asyncio
    async def test_active_count_tracks_connections(self):
        from api.websocket_manager import WebSocketManager
        from unittest.mock import AsyncMock, MagicMock

        manager = WebSocketManager()
        assert manager.active_count == 0

        # Mock WebSocket
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()

        await manager.connect(
            connection_id="test-conn-1",
            websocket=mock_ws,
            user_id="user-abc",
            role="GUARD",
        )
        assert manager.active_count == 1

        await manager.disconnect("test-conn-1")
        assert manager.active_count == 0

    @pytest.mark.asyncio
    async def test_multiple_connections_all_receive_broadcast(self):
        from api.websocket_manager import WebSocketManager
        from unittest.mock import AsyncMock, MagicMock
        from starlette.websockets import WebSocketState

        manager = WebSocketManager()
        received = {f"conn-{i}": [] for i in range(3)}

        for i in range(3):
            mock_ws = MagicMock()
            mock_ws.accept = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.send_text = AsyncMock(
                side_effect=lambda msg, i=i: received[f"conn-{i}"].append(msg)
            )
            await manager.connect(
                connection_id=f"conn-{i}",
                websocket=mock_ws,
                user_id=f"user-{i}",
                role="GUARD",
            )

        test_message = {"event_type": "TEST", "payload": {"value": 42}}
        await manager.broadcast(test_message)

        for conn_id, messages in received.items():
            assert len(messages) == 1, f"{conn_id} should have received 1 message"
            parsed = json.loads(messages[0])
            assert parsed["payload"]["value"] == 42
