from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ResponseState:
    client_type: str = "unknown"
    purchase_stage: str = "unknown"
    last_question_type: str = "unknown"
    dialog_phase: str = "discovery"
    conversation_closed: bool = False
    greeted: bool = False
    active_flow: str = "none"
    flow_step: str = "idle"
    last_action_name: str = ""
    same_action_repeats: int = 0
    manager_handoff_stage: str = "none"
    evidence_status: str = "unknown"
    active_business_flow: str = "none"
    slots: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "ResponseState":
        repeats_raw = snapshot.get("same_action_repeats", 0)
        try:
            repeats = int(repeats_raw)
        except (TypeError, ValueError):
            repeats = 0
        return cls(
            client_type=str(snapshot.get("client_type", "unknown")),
            purchase_stage=str(snapshot.get("purchase_stage", "unknown")),
            last_question_type=str(snapshot.get("last_question_type", "unknown")),
            dialog_phase=str(snapshot.get("dialog_phase", "discovery")),
            conversation_closed=bool(snapshot.get("conversation_closed", False)),
            greeted=bool(snapshot.get("greeted", False)),
            active_flow=str(snapshot.get("active_flow", "none")),
            flow_step=str(snapshot.get("flow_step", "idle")),
            last_action_name=str(snapshot.get("last_action_name", "")),
            same_action_repeats=repeats,
            manager_handoff_stage=str(snapshot.get("manager_handoff_stage", "none")),
            evidence_status=str(snapshot.get("evidence_status", "unknown")),
            active_business_flow=str(snapshot.get("active_business_flow", "none")),
            slots=dict(snapshot.get("slots", {})) if isinstance(snapshot.get("slots", {}), dict) else {},
        )


@dataclass(slots=True)
class ResponseAction:
    name: str
    primary_topic: str
    secondary_topic: str | None = None
    locked_action: bool = False
