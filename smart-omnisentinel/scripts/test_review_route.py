import asyncio
import uuid
from fastapi.testclient import TestClient
from api.main import app
from db.session import get_session_factory
from db.models.user import User
from db.models.incident import Incident

client = TestClient(app)

async def run():
    factory = get_session_factory()
    async with factory() as db:
        # get an active incident
        result = await db.execute(__import__("sqlalchemy").text("SELECT id FROM incidents LIMIT 1"))
        iid = result.scalar_one_or_none()
        
    print(f"Testing incident: {iid}")
    
    # login
    resp = client.post("/api/v1/auth/login", json={"email": "admin@demo.com", "password": "Admin123!"})
    token = resp.json()["access_token"]
    
    # trigger review
    resp2 = client.post(
        f"/api/v1/review/{iid}/action",
        json={"action": "FALSE_POSITIVE", "notes": "test"},
        headers={"Authorization": f"Bearer {token}"}
    )
    print(resp2.status_code)
    print(resp2.text)

asyncio.run(run())
