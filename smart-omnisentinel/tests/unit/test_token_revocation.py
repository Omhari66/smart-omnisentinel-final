"""
tests/unit/test_token_revocation.py
-------------------------------------
Unit tests for JWT token revocation (Phase 5).
Tests the jti claim, in-process blocklist, and revoke_token() behaviour.
All tests run without Redis (REDIS__ENABLED=false via conftest.py test_env fixture).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pytest

from core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    revoke_token,
    _in_process_blocklist,
)
from core.exceptions import AuthenticationError


@pytest.fixture(autouse=True)
def clear_blocklist():
    """Clear the in-process blocklist before and after each test."""
    _in_process_blocklist.clear()
    yield
    _in_process_blocklist.clear()


class TestJtiClaim:

    def test_access_token_has_jti(self):
        token = create_access_token(subject="user-123", role="operator")
        payload = decode_token(token)
        assert "jti" in payload
        assert len(payload["jti"]) == 36   # UUID4 format

    def test_refresh_token_has_jti(self):
        token = create_refresh_token(subject="user-123")
        payload = decode_token(token)
        assert "jti" in payload
        assert len(payload["jti"]) == 36

    def test_each_token_has_unique_jti(self):
        """Two tokens for the same user must have different jtis."""
        token1 = create_access_token(subject="user-123", role="operator")
        token2 = create_access_token(subject="user-123", role="operator")
        jti1 = decode_token(token1)["jti"]
        jti2 = decode_token(token2)["jti"]
        assert jti1 != jti2

    def test_token_type_claim_preserved(self):
        access = create_access_token(subject="user-1", role="admin")
        refresh = create_refresh_token(subject="user-1")
        assert decode_token(access)["type"] == "access"
        assert decode_token(refresh)["type"] == "refresh"


class TestRevocation:

    @pytest.mark.asyncio
    async def test_revoked_token_raises_auth_error(self):
        token = create_access_token(subject="user-123", role="operator")
        payload = decode_token(token)
        jti = payload["jti"]
        exp = payload["exp"]
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)

        # Token valid before revocation
        result = decode_token(token)
        assert result["sub"] == "user-123"

        # Revoke it
        await revoke_token(jti, expires_at)

        # Now it must be rejected
        with pytest.raises(AuthenticationError, match="revoked"):
            decode_token(token)

    @pytest.mark.asyncio
    async def test_revoke_adds_jti_to_blocklist(self):
        jti = "test-jti-abc123"
        expires_at = datetime.now(tz=timezone.utc) + timedelta(minutes=15)

        assert jti not in _in_process_blocklist
        await revoke_token(jti, expires_at)
        assert jti in _in_process_blocklist

    @pytest.mark.asyncio
    async def test_different_token_not_affected_by_revocation(self):
        token1 = create_access_token(subject="user-1", role="operator")
        token2 = create_access_token(subject="user-1", role="operator")

        payload1 = decode_token(token1)
        expires_at = datetime.fromtimestamp(payload1["exp"], tz=timezone.utc)

        # Revoke only token1
        await revoke_token(payload1["jti"], expires_at)

        # token2 must still work
        result = decode_token(token2)
        assert result["sub"] == "user-1"

    @pytest.mark.asyncio
    async def test_already_expired_token_revocation_harmless(self):
        """
        Revoking an already-expired jti should not raise an error.
        TTL will be 0 or negative — the Redis entry won't be stored but
        the in-process set will just hold the jti until cleared.
        """
        past = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
        await revoke_token("already-expired-jti", past)
        # No exception raised — passes
