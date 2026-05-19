from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.models import TopicClassificationResult


@dataclass(slots=True)
class TurnAnalysis:
    current_focus: str = "unknown"
    slots: dict[str, object] = field(default_factory=dict)
    nonsense_input: bool = False
    abuse_message: bool = False
    pricing_request: bool = False
    catalog_list_request: bool = False
    feature_comparison: bool = False
    refund_policy: bool = False
    post_payment_access: bool = False
    multi_user_access: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TurnAnalysis":
        data = payload or {}
        slots_raw = data.get("slots", {})
        slots = dict(slots_raw) if isinstance(slots_raw, dict) else {}
        return cls(
            current_focus=str(data.get("current_focus", "unknown")).strip().lower() or "unknown",
            slots=slots,
            nonsense_input=bool(data.get("nonsense_input", False)),
            abuse_message=bool(data.get("abuse_message", False)),
            pricing_request=bool(data.get("pricing_request", False)),
            catalog_list_request=bool(data.get("catalog_list_request", False)),
            feature_comparison=bool(data.get("feature_comparison", False)),
            refund_policy=bool(data.get("refund_policy", False)),
            post_payment_access=bool(data.get("post_payment_access", False)),
            multi_user_access=bool(data.get("multi_user_access", False)),
        )


def extract_turn_analysis(topic_result: TopicClassificationResult) -> TurnAnalysis:
    raw = topic_result.diagnostics.get("turn_analysis", {})
    data = raw if isinstance(raw, dict) else {}
    return TurnAnalysis.from_dict(data)

