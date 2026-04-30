"""
scripts/generate_test_incident.py
------------------------------------
Inject a synthetic incident into the system for UI/dashboard testing.
Creates a real DB record + pushes a WebSocket event.
Useful for testing the dashboard without a live camera.

Usage:
    python scripts/generate_test_incident.py \
        --camera_id <uuid> \
        --severity HIGH \
        --event_type VIOLENT_INTERACTION
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def inject_test_incident(
    camera_id: str,
    severity: str,
    event_type: str,
    risk_score: int,
) -> None:
    from core.config import get_settings
    from core.logger import setup_logging
    setup_logging()

    from db.session import get_session_factory
    from db.models.incident import Incident

    factory = get_session_factory()
    async with factory() as db:
        incident = Incident(
            camera_id=uuid.UUID(camera_id),
            event_type=event_type,
            severity=severity,
            status="PENDING_REVIEW" if severity == "LOW" else
                   "ACTIVE_REVIEW" if severity == "MEDIUM" else "EMERGENCY",
            risk_score=risk_score,
            confidence_at_alert=0.87,
            people_count=2,
            event_start=datetime.now(tz=timezone.utc),
            detected_at=datetime.now(tz=timezone.utc),
            is_locked=(severity == "HIGH"),
            retention_days=3 if severity == "LOW" else 30 if severity == "MEDIUM" else 365,
        )
        db.add(incident)
        await db.commit()
        await db.refresh(incident)
        print(f"✓ Test incident created: {incident.id}")
        print(f"  Type:     {event_type}")
        print(f"  Severity: {severity}")
        print(f"  Risk:     {risk_score}")
        return str(incident.id)


def main():
    parser = argparse.ArgumentParser(description="Inject test incident for UI testing")
    parser.add_argument("--camera_id", required=True, help="Camera UUID")
    parser.add_argument("--severity", default="MEDIUM",
                        choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--event_type", default="VIOLENT_INTERACTION",
                        choices=["VIOLENT_INTERACTION", "FALL_COLLAPSE",
                                 "CROWD_AGGRESSION", "PERSON_DISTRESS"])
    parser.add_argument("--risk_score", type=int, default=55)
    args = parser.parse_args()

    asyncio.run(inject_test_incident(
        camera_id=args.camera_id,
        severity=args.severity,
        event_type=args.event_type,
        risk_score=args.risk_score,
    ))


if __name__ == "__main__":
    main()
