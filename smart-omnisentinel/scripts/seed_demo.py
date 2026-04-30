"""
scripts/seed_demo.py
---------------------
Seeds demo.db with:
  - 3 cameras (Main Entrance, Parking Lot, Stairwell)
  - 5 pre-built incidents at LOW / MEDIUM / HIGH severities
  - Ensures admin user exists

Run once before starting the demo:
    python scripts/seed_demo.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def seed() -> None:
    # Import here so path is set first
    from sqlalchemy import select, text
    from db.session import get_session_factory, get_engine
    from db.base import Base
    import db.models  # registers all models

    from db.models.camera import Camera
    from db.models.incident import Incident
    from db.models.user import User
    from core.constants import CameraStatus, EventType, IncidentStatus, Severity
    from core.security import hash_password

    engine = get_engine()

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓  Tables verified / created")

    factory = get_session_factory()
    async with factory() as db:

        # ── Admin user ────────────────────────────────────────────────────
        existing_admin = (await db.execute(
            select(User).where(User.email == "admin@demo.com")
        )).scalar_one_or_none()

        if not existing_admin:
            db.add(User(
                email="admin@demo.com",
                hashed_password=hash_password("Admin123!"),
                full_name="Administrator",
                role="ADMIN",
                is_active=True,
            ))
            await db.flush()
            print("✓  Admin user created  (admin@demo.com / Admin123!)")
        else:
            print("✓  Admin user already exists")

        # ── Cameras ───────────────────────────────────────────────────────
        cam_defs = [
            ("Main Entrance",  "rtsp://192.168.1.10/stream1", "Building A – Front Door",  CameraStatus.ACTIVE),
            ("Parking Lot B",  "rtsp://192.168.1.11/stream1", "Outdoor – North Lot",       CameraStatus.ACTIVE),
            ("Stairwell C3",   "rtsp://192.168.1.12/stream1", "Building C – Floor 3",      CameraStatus.OFFLINE),
        ]

        cam_ids: list[uuid.UUID] = []
        now = datetime.now(tz=timezone.utc)

        for name, url, loc, status in cam_defs:
            existing = (await db.execute(
                select(Camera).where(Camera.name == name)
            )).scalar_one_or_none()

            if existing:
                cam_ids.append(existing.id)
                print(f"  ↳ Camera '{name}' already exists")
            else:
                cam = Camera(
                    name=name,
                    stream_url=url,
                    location=loc,
                    status=status,
                    last_seen_at=now if status == CameraStatus.ACTIVE else None,
                    fps_target=4,
                )
                db.add(cam)
                await db.flush()
                cam_ids.append(cam.id)
                print(f"  ↳ Camera '{name}' created")

        print(f"✓  {len(cam_ids)} cameras ready")

        # ── Pre-built incidents ───────────────────────────────────────────
        incident_defs = [
            # (camera_idx, event_type, severity, status, risk, conf, people, mins_ago, duration)
            (0, EventType.VIOLENT_INTERACTION, Severity.HIGH,   IncidentStatus.EMERGENCY,      82, 0.87, 2, 45, 12.0),
            (1, EventType.CROWD_AGGRESSION,    Severity.MEDIUM, IncidentStatus.ACTIVE_REVIEW,   55, 0.71, 6, 20,  8.0),
            (0, EventType.FALL_COLLAPSE,       Severity.HIGH,   IncidentStatus.PENDING_REVIEW,  78, 0.93, 1, 10,  5.0),
            (1, EventType.VIOLENT_INTERACTION, Severity.LOW,    IncidentStatus.RESOLVED,        22, 0.52, 2,  5,  3.0),
            (0, EventType.VIOLENT_INTERACTION, Severity.MEDIUM, IncidentStatus.PENDING_REVIEW,  61, 0.68, 3,  2,  6.5),
        ]

        existing_count = (await db.execute(
            select(Incident)
        )).scalars().all()

        if len(existing_count) >= len(incident_defs):
            print(f"✓  Incidents already seeded ({len(existing_count)} found)")
        else:
            for (ci, etype, sev, status, risk, conf, people, mins_ago, dur) in incident_defs:
                t_start = now - timedelta(minutes=mins_ago)
                db.add(Incident(
                    camera_id=cam_ids[ci],
                    event_type=etype,
                    severity=sev,
                    status=status,
                    risk_score=risk,
                    confidence_at_alert=conf,
                    people_count=people,
                    event_start=t_start,
                    event_end=t_start + timedelta(seconds=dur),
                    detected_at=t_start,
                    duration_seconds=dur,
                    is_locked=False,
                    retention_days=30,
                ))
            await db.flush()
            print(f"✓  {len(incident_defs)} incidents seeded")

        await db.commit()

    print()
    print("=" * 50)
    print("  Demo data ready!")
    print("  Login: admin@demo.com  /  Admin123!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(seed())
