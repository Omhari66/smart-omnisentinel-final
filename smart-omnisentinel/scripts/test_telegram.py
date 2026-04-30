import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings
import logging

logging.basicConfig(level=logging.INFO)

settings = get_settings()

print(f"Telegram Enabled: {settings.notifications.telegram_enabled}")
print(f"Bot Token: {settings.notifications.telegram_bot_token[:10]}...")
print(f"Chat ID: {settings.notifications.telegram_chat_id}")

# Attempt to send a basic message
if settings.notifications.telegram_enabled and settings.notifications.telegram_bot_token:
    import requests
    url = f"https://api.telegram.org/bot{settings.notifications.telegram_bot_token}/sendMessage"
    payload = {"chat_id": settings.notifications.telegram_chat_id, "text": "Test from SmartOmniSentinel Test Script"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Telegram is NOT enabled in .env or token is missing.")
