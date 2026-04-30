import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from db.session import get_session_factory
from db.models.incident import Incident
from services.alert_manager.notifier import NotificationService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dispatch():
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(Incident).limit(1))
        incident = result.scalar_one_or_none()
        
        if not incident:
            print("No incident found in DB.")
            return

        print(f"Testing dispatch for Incident ID: {incident.id}")
        
        class DummyIncident:
            id = incident.id
            event_type = incident.event_type
            risk_score = incident.risk_score
            detected_at = incident.detected_at
            class DummyCamera:
                name = "Test Camera"
                location = "Test Location"
            camera = DummyCamera()

        notifier = NotificationService()
        notifier.dispatch_emergency_alert(DummyIncident())
        print("Dispatch function completed.")

asyncio.run(test_dispatch())
