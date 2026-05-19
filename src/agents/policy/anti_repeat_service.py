from __future__ import annotations

from pathlib import Path

from src.agents.policy.anti_repeat_policy import ResponseAntiRepeatPolicy
from src.agents.policy.types import ResponseAction


class AntiRepeatService:
    """Dedicated anti-repeat postprocessor service."""

    def __init__(self, response_policy_file_path: Path) -> None:
        self._policy = ResponseAntiRepeatPolicy(response_policy_file_path)

    def apply(
        self,
        action: ResponseAction,
        answer_text: str,
        history_text: str,
        user_query: str,
    ) -> str:
        return self._policy.apply(
            action=action,
            answer_text=answer_text,
            history_text=history_text,
            user_query=user_query,
        )
