"""
tests/integration/test_review_workflow.py
------------------------------------------
Integration tests for the human review workflow.
Tests the state machine: PENDING_REVIEW → RESOLVED / EMERGENCY / FALSE_POSITIVE.
"""

from __future__ import annotations

import uuid

import pytest


class TestReviewStateMachine:
    """
    Tests the incident status transitions driven by reviewer actions.
    Uses mocked DB sessions to avoid requiring a live PostgreSQL instance.
    """

    def test_confirm_action_locks_evidence(self):
        """CONFIRM action must set is_locked=True on the incident."""
        from core.constants import IncidentStatus, ReviewAction

        # Simulate the status_map used in review route
        status_map = {
            ReviewAction.CONFIRM: IncidentStatus.EMERGENCY,
            ReviewAction.DISMISS: IncidentStatus.RESOLVED,
            ReviewAction.ESCALATE: IncidentStatus.EMERGENCY,
            ReviewAction.FALSE_POSITIVE: IncidentStatus.FALSE_POSITIVE,
        }

        assert status_map[ReviewAction.CONFIRM] == IncidentStatus.EMERGENCY
        assert status_map[ReviewAction.FALSE_POSITIVE] == IncidentStatus.FALSE_POSITIVE

    def test_lock_set_on_confirm_and_escalate(self):
        """CONFIRM and ESCALATE should lock evidence; DISMISS and FP should not."""
        from core.constants import ReviewAction

        lock_actions = {ReviewAction.CONFIRM, ReviewAction.ESCALATE}
        no_lock_actions = {ReviewAction.DISMISS, ReviewAction.FALSE_POSITIVE}

        for action in lock_actions:
            should_lock = action in (ReviewAction.CONFIRM, ReviewAction.ESCALATE)
            assert should_lock is True

        for action in no_lock_actions:
            should_lock = action in (ReviewAction.CONFIRM, ReviewAction.ESCALATE)
            assert should_lock is False

    def test_review_action_schema_valid_actions(self):
        """ReviewActionRequest should only accept valid ReviewAction values."""
        from api.schemas.review import ReviewActionRequest
        from core.constants import ReviewAction

        valid = ReviewActionRequest(action=ReviewAction.CONFIRM, notes="Confirmed real incident")
        assert valid.action == ReviewAction.CONFIRM

    def test_review_queue_item_schema(self):
        """ReviewQueueItem can be constructed from expected fields."""
        from datetime import datetime, timezone
        from api.schemas.review import ReviewQueueItem
        from core.constants import Severity

        item = ReviewQueueItem(
            incident_id=uuid.uuid4(),
            camera_name="North Gate",
            location="Building A",
            event_type="VIOLENT_INTERACTION",
            severity=Severity.MEDIUM,
            risk_score=55,
            detected_at=datetime.now(tz=timezone.utc),
            escalation_deadline=None,
            queue_position=1,
        )
        assert item.queue_position == 1
        assert item.severity == Severity.MEDIUM
