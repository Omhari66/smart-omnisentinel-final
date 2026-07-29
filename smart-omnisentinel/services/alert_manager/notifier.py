"""
services/alert_manager/notifier.py
----------------------------------
Handles dispatching real SMS and Email notifications using Twilio and SMTP.
If credentials are not provided, it falls back to a simulated "Mock Mode"
to allow demonstrations without API keys.
"""

from __future__ import annotations

import logging
from typing import Optional
from smtplib import SMTP_SSL
from email.message import EmailMessage

from core.config import get_settings
from db.models.incident import Incident

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        self.settings = get_settings()

    def dispatch_emergency_alert(self, incident: Incident) -> None:
        """
        Master function to format and send alerts via all configured channels.
        """
        camera_name = incident.camera.name if incident.camera else "Unknown Camera"
        location = incident.camera.location if incident.camera else "Unknown Location"
        
        ai_summary = f"\nAI Summary: {incident.notes}\n" if incident.notes else ""
        
        message = (
            f"🚨 EMERGENCY ALERT: {incident.event_type}\n"
            f"Location: {camera_name} ({location})\n"
            f"Risk Score: {incident.risk_score}/100\n"
            f"Time: {incident.detected_at.strftime('%H:%M:%S')} UTC\n"
            f"{ai_summary}"
            f"Action Required: Immediate Review."
        )

        logger.info(f"Preparing to dispatch alerts for Incident {incident.id}")

        # Dispatch SMS
        self._send_sms(message)

        # Dispatch Email
        subject = f"🚨 URGENT: Security Alert at {camera_name}"
        self._send_email(subject, message)
        
        # Dispatch Telegram
        self._send_telegram(incident, message)

    def _send_telegram(self, incident: Incident, message: str) -> None:
        token = self.settings.notifications.telegram_bot_token
        chat_id = self.settings.notifications.telegram_chat_id

        if not self.settings.notifications.telegram_enabled or not token or token == "YOUR_BOT_TOKEN":
            logger.info(f"[MOCK TELEGRAM] Would send to chat {chat_id}: {message}")
            return

        try:
            import requests
            from pathlib import Path
            import glob

            # Find the video clip for this incident
            clip_dir = Path("evidence_storage/clips")
            clip_pattern = f"{str(incident.id)}_*.mp4"
            clip_matches = glob.glob(str(clip_dir / clip_pattern))
            
            # Find the snapshot for this incident
            evidence_dir = Path("evidence_storage/snapshots")
            search_pattern = f"{str(incident.id)[:8]}_*.jpg"
            snap_matches = glob.glob(str(evidence_dir / search_pattern))
            
            payload = {
                "chat_id": chat_id,
                "caption": message
            }

            # Try Video First
            if clip_matches:
                url = f"https://api.telegram.org/bot{token}/sendVideo"
                latest_clip = sorted(clip_matches)[-1]
                with open(latest_clip, "rb") as video:
                    files = {"video": video}
                    response = requests.post(url, data=payload, files=files, timeout=45)
            # Fallback to Photo
            elif snap_matches:
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                latest_snapshot = sorted(snap_matches)[-1]
                with open(latest_snapshot, "rb") as photo:
                    files = {"photo": photo}
                    response = requests.post(url, data=payload, files=files, timeout=10)
            # Fallback to pure Text
            else:
                msg_url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {"chat_id": chat_id, "text": message}
                response = requests.post(msg_url, json=payload, timeout=10)

            response.raise_for_status()
            logger.info("Telegram alert sent successfully!")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def _send_sms(self, message: str) -> None:
        sid = self.settings.notifications.twilio_account_sid
        token = self.settings.notifications.twilio_auth_token
        from_num = self.settings.notifications.twilio_from_number
        to_numbers = self.settings.notifications.twilio_to_numbers
        
        # Determine the target number (use the first one in the list or a default mock one)
        to_num = to_numbers[0] if to_numbers else "+15551111111"

        if not self.settings.notifications.twilio_enabled or not sid or sid == "YOUR_TWILIO_SID":
            logger.info(f"[MOCK SMS] Would send to {to_num}: {message}")
            return

        try:
            from twilio.rest import Client
            client = Client(sid, token)
            msg = client.messages.create(
                body=message,
                from_=from_num,
                to=to_num
            )
            logger.info(f"Twilio SMS sent successfully! SID: {msg.sid}")
        except ImportError:
            logger.error("Twilio package not installed. Run: pip install twilio")
        except Exception as e:
            logger.error(f"Failed to send Twilio SMS: {e}")

    def _send_email(self, subject: str, body: str) -> None:
        smtp_server = self.settings.notifications.smtp_host
        smtp_port = self.settings.notifications.smtp_port
        smtp_user = self.settings.notifications.smtp_user
        smtp_pass = self.settings.notifications.smtp_password
        to_emails = self.settings.notifications.alert_recipients
        to_email = to_emails[0] if to_emails else "security@demo.com"

        if not smtp_server or smtp_server == "smtp.example.com":
            logger.info(f"[MOCK EMAIL] Would send to {to_email}: Subject: {subject}")
            return

        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = to_email

            with SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            logger.info("Alert Email sent successfully!")
        except Exception as e:
            logger.error(f"Failed to send Email: {e}")
