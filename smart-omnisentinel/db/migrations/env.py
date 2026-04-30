"""db/migrations/env.py - Alembic environment supporting both PostgreSQL and SQLite."""
from __future__ import annotations
import asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from logging.config import fileConfig
from alembic import context
from db.base import Base
import db.models  # noqa: F401
from core.config import get_settings

config = context.config
settings = get_settings()
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata

def get_url():
    url = settings.db.url
    # Convert async URL to sync for Alembic
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("sqlite+aiosqlite://", "sqlite:///")
    return url

def run_migrations_offline():
    context.configure(url=get_url(), target_metadata=target_metadata,
                      literal_binds=True, compare_type=True,
                      render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata,
                      compare_type=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    from sqlalchemy import create_engine
    connectable = create_engine(get_url())
    with connectable.connect() as connection:
        do_run_migrations(connection)

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()