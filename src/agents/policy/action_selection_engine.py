from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.policy.types import ResponseAction, ResponseState
from src.agents.response_policy import ResponseActionSelector


class ActionSelectionPolicyEngine:
    """Policy-engine facade over response action selection rules."""

    def __init__(self, response_policy_file_path: Path) -> None:
        self._selector = ResponseActionSelector(response_policy_file_path)

    def select_from_planner(
        self,
        topic_ids: list[str],
        planned_action: str,
        current_focus: str,
    ) -> ResponseAction | None:
        return self._selector.select_from_planner(
            topic_ids=topic_ids,
            planned_action=planned_action,
            current_focus=current_focus,
        )

    def select(
        self,
        topic_ids: list[str],
        state: ResponseState,
        user_query: str,
        history_text: str,
        turn_analysis: dict[str, Any] | None = None,
    ) -> ResponseAction:
        return self._selector.select(
            topic_ids=topic_ids,
            state=state,
            user_query=user_query,
            history_text=history_text,
            turn_analysis=turn_analysis,
        )
