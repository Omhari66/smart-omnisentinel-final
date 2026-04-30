"""
core/exceptions.py
------------------
Custom exception classes.
Raised in service layer; caught in API middleware for consistent error responses.
"""


class SentinelBaseError(Exception):
    """Root exception for all application errors."""
    status_code: int = 500
    detail: str = "An internal error occurred."

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


# ---------------------------------------------------------------------------
# 4xx Client Errors
# ---------------------------------------------------------------------------

class NotFoundError(SentinelBaseError):
    status_code = 404
    detail = "Resource not found."


class AlreadyExistsError(SentinelBaseError):
    status_code = 409
    detail = "Resource already exists."


class ValidationError(SentinelBaseError):
    status_code = 422
    detail = "Validation failed."


class AuthenticationError(SentinelBaseError):
    status_code = 401
    detail = "Authentication required."


class AuthorizationError(SentinelBaseError):
    status_code = 403
    detail = "Insufficient permissions."


class EvidenceLockedError(SentinelBaseError):
    status_code = 403
    detail = "Evidence clip is locked and cannot be modified or deleted."


class CooldownActiveError(SentinelBaseError):
    status_code = 429
    detail = "Alert cooldown is active for this camera and event type."


# ---------------------------------------------------------------------------
# 5xx Server / Infrastructure Errors
# ---------------------------------------------------------------------------

class StreamConnectionError(SentinelBaseError):
    status_code = 503
    detail = "Failed to connect to camera stream."


class InferenceError(SentinelBaseError):
    status_code = 500
    detail = "Inference pipeline encountered an error."


class StorageError(SentinelBaseError):
    status_code = 500
    detail = "Evidence storage operation failed."


class NotificationError(SentinelBaseError):
    status_code = 500
    detail = "Notification dispatch failed."


class DatabaseError(SentinelBaseError):
    status_code = 500
    detail = "Database operation failed."


class ConfigurationError(SentinelBaseError):
    status_code = 500
    detail = "System configuration error."
