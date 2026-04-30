"""
core/config.py
--------------
Single source of truth for all runtime configuration.
Loaded once at startup; injected via FastAPI dependency where needed.

Override any value via environment variables using double-underscore
nesting notation:  INFERENCE__DEVICE=cuda
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    url: str = "postgresql+asyncpg://sentinel:sentinel_pass@localhost:5432/sentinel"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False  # Set True only in dev to log SQL


class RedisSettings(BaseSettings):
    url: str = "redis://localhost:6379/0"
    enabled: bool = False  # False → use in-process asyncio.Queue (MVP mode)


class JWTSettings(BaseSettings):
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7


class InferenceSettings(BaseSettings):
    device: str = "cpu"                  # "cpu" | "cuda"
    detector_model: str = "yolov8n.pt"
    pose_model: str = "yolov8n-pose.pt"
    classifier_checkpoint: str = "ml/checkpoints/temporal_v1.pt"
    effective_fps: int = 4               # Effective sampling rate after frame skip
    detection_skip_frames: int = 4       # Run detector every Nth frame
    persistence_frames: int = 12         # Frames event must persist before confirming
    ema_alpha: float = 0.35              # EMA smoothing factor
    use_tensorrt: bool = False           # Enable TensorRT path on Jetson


class StorageSettings(BaseSettings):
    evidence_root: str = "evidence_storage"
    max_disk_gb: float = 50.0
    staging_ttl_hours: int = 2           # Auto-clean staging clips older than this


class NotificationSettings(BaseSettings):
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "SmartOmniSentinel <alerts@yoursite.com>"
    alert_recipients: List[str] = Field(default_factory=list)
    webhook_urls: List[str] = Field(default_factory=list)
    twilio_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_numbers: List[str] = Field(default_factory=list)
    
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


class SignedURLSettings(BaseSettings):
    secret: str = "change-me-signed-url-secret"
    expiry_seconds: int = 900


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Top-level settings
    app_name: str = "SmartOmniSentinel"
    app_env: str = "development"
    debug: bool = False
    secret_key: str = "CHANGE-ME-IN-PRODUCTION-USE-LONG-RANDOM-STRING"
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )

    # Nested settings groups
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    signed_url: SignedURLSettings = Field(default_factory=SignedURLSettings)

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "production", "testing"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "testing"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Call get_settings() anywhere; it reads .env once and caches.
    In tests, clear the cache with get_settings.cache_clear().
    """
    return Settings()
