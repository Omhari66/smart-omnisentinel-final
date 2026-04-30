"""
tests/conftest.py
-----------------
Shared pytest fixtures and configuration.
"""

import asyncio
import os
import sys

import pytest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for all async tests in the session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def test_env(monkeypatch):
    """Set test environment variables for all tests."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("SECRET_KEY", "test-secret-at-least-32-characters-long")
    monkeypatch.setenv("DB__URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("REDIS__ENABLED", "false")
    monkeypatch.setenv("STORAGE__EVIDENCE_ROOT", "/tmp/test_evidence")

    # Clear settings cache so test env vars take effect
    from core.config import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()
