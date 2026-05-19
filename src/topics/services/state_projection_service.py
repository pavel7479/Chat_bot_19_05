from __future__ import annotations

from typing import Any

from src.topics.dto import StateProjection


class StateProjectionService:
    """Projects state snapshots for diagnostics/telemetry contracts."""

    def project(self, state_before: Any, state_after: Any) -> StateProjection:
        before = {
            "client_type": getattr(state_before, "client_type", "unknown"),
            "purchase_stage": getattr(state_before, "purchase_stage", "unknown"),
            "last_question_type": getattr(state_before, "last_question_type", "unknown"),
            "short_reply_polarity": getattr(state_before, "short_reply_polarity", "unknown"),
            "dialog_phase": getattr(state_before, "dialog_phase", "discovery"),
            "conversation_closed": bool(getattr(state_before, "conversation_closed", False)),
            "greeted": bool(getattr(state_before, "greeted", False)),
        }
        after = state_after.as_dict() if hasattr(state_after, "as_dict") else {}
        return StateProjection(state_before=before, state_after=after)
