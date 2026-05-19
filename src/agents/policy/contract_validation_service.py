from __future__ import annotations

from pathlib import Path

from src.agents.policy.types import ResponseAction
from src.agents.response_policy import ResponseContractValidator


class ContractValidationService:
    """Contract validation service for action-response consistency."""

    def __init__(self, response_policy_file_path: Path) -> None:
        self._validator = ResponseContractValidator(response_policy_file_path)

    def validate(self, action: ResponseAction, answer_text: str) -> bool:
        return self._validator.validate(action, answer_text)
