from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FSMDecision:
    state: str
    planned_action: str


class FSMOrchestrator:
    def __init__(self, spec: dict[str, Any]) -> None:
        transitions_raw = spec.get("transitions", []) if isinstance(spec, dict) else []
        self._transitions: list[tuple[str, str, str, str]] = []
        for item in transitions_raw:
            if not isinstance(item, dict):
                continue
            state = str(item.get("state", "*")).strip() or "*"
            intent = str(item.get("intent", "")).strip()
            next_state = str(item.get("next_state", state)).strip() or state
            planned_action = str(item.get("planned_action", "clarify_request")).strip() or "clarify_request"
            if intent:
                self._transitions.append((state, intent, next_state, planned_action))

    def transition(self, state: str, intent: str) -> FSMDecision:
        current = state.strip() or "discovery"
        for allowed_state, allowed_intent, next_state, action in self._transitions:
            if allowed_intent != intent:
                continue
            if allowed_state != "*" and allowed_state != current:
                continue
            return FSMDecision(state=next_state, planned_action=action)
        return FSMDecision(state=current, planned_action="clarify_request")
