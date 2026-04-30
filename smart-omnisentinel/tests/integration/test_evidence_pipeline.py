"""
tests/integration/test_evidence_pipeline.py
---------------------------------------------
Integration tests for the evidence clip pipeline.
Tests: SHA-256 hashing, retention policy logic, signed URL generation.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid

import pytest


class TestIntegrityHasher:

    def test_sha256_consistent(self):
        """Same file always produces the same hash."""
        from services.evidence_manager.integrity_hasher import compute_sha256

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake video content for testing " * 100)
            path = f.name

        try:
            h1 = compute_sha256(path)
            h2 = compute_sha256(path)
            assert h1 is not None
            assert h1 == h2
            assert len(h1) == 64  # SHA-256 hex digest length
        finally:
            os.unlink(path)

    def test_sha256_detects_tampering(self):
        """Modified file produces a different hash."""
        from services.evidence_manager.integrity_hasher import compute_sha256, verify_sha256

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"original content " * 50)
            path = f.name

        try:
            original_hash = compute_sha256(path)
            assert original_hash is not None

            # Tamper with file
            with open(path, "ab") as f:
                f.write(b"tampered")

            # Verification should fail
            assert not verify_sha256(path, original_hash)
        finally:
            os.unlink(path)

    def test_sha256_missing_file_returns_none(self):
        from services.evidence_manager.integrity_hasher import compute_sha256
        result = compute_sha256("/nonexistent/path/file.mp4")
        assert result is None

    def test_verify_sha256_correct_hash(self):
        from services.evidence_manager.integrity_hasher import compute_sha256, verify_sha256

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            content = b"test evidence content"
            f.write(content)
            path = f.name

        try:
            expected = hashlib.sha256(content).hexdigest()
            assert verify_sha256(path, expected)
        finally:
            os.unlink(path)


class TestSignedURL:

    def test_generate_and_verify_valid_url(self):
        from core.signed_url import generate_signed_url, verify_signed_url
        import time

        path = f"/evidence/clips/test-{uuid.uuid4()}_full.mp4"
        signed = generate_signed_url(
            base_url="http://localhost:8000/api/v1",
            path=path,
            expiry_seconds=3600,
        )

        assert "token=" in signed
        assert "expires=" in signed

        # Parse params and verify
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(signed)
        params = parse_qs(parsed.query)
        token = params["token"][0]
        expires = int(params["expires"][0])

        # Should not raise
        verify_signed_url(path=path, token=token, expires=expires)

    def test_expired_url_raises(self):
        from core.signed_url import generate_signed_url, verify_signed_url
        from core.exceptions import AuthenticationError
        import time

        path = "/evidence/clips/expired-test.mp4"
        # Generate with 1-second expiry
        signed = generate_signed_url(
            base_url="http://localhost:8000",
            path=path,
            expiry_seconds=1,
        )
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(signed)
        params = parse_qs(parsed.query)
        token = params["token"][0]
        expires = int(params["expires"][0])

        time.sleep(1.1)

        with pytest.raises(AuthenticationError, match="expired"):
            verify_signed_url(path=path, token=token, expires=expires)

    def test_wrong_token_raises(self):
        from core.signed_url import generate_signed_url, verify_signed_url
        from core.exceptions import AuthenticationError

        path = "/evidence/clips/tampered.mp4"
        signed = generate_signed_url(
            base_url="http://localhost:8000",
            path=path,
            expiry_seconds=3600,
        )
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(signed)
        params = parse_qs(parsed.query)
        expires = int(params["expires"][0])

        with pytest.raises(AuthenticationError):
            verify_signed_url(path=path, token="wrong_token_abc123", expires=expires)

    def test_tampered_path_raises(self):
        from core.signed_url import generate_signed_url, verify_signed_url
        from core.exceptions import AuthenticationError

        path = "/evidence/clips/original.mp4"
        tampered_path = "/evidence/clips/different.mp4"
        signed = generate_signed_url(
            base_url="http://localhost:8000",
            path=path,
            expiry_seconds=3600,
        )
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(signed)
        params = parse_qs(parsed.query)
        token = params["token"][0]
        expires = int(params["expires"][0])

        with pytest.raises(AuthenticationError):
            verify_signed_url(path=tampered_path, token=token, expires=expires)
