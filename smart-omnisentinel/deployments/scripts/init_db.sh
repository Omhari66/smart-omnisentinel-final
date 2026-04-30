#!/usr/bin/env bash
set -euo pipefail
echo "SmartOmniSentinel — Database Init"
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); echo "Loaded .env"; fi
echo "Running Alembic migrations..."
alembic upgrade head
echo "Migrations complete. Run seed_demo_data.sh for demo data."
