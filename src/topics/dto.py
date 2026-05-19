from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class TurnSignals:
    signals: list[str]
    locked_topics: list[str]
    slots: dict[str, bool]
    current_focus: str
    docs_query: bool
    services_overview_query: bool
    out_of_scope_query: bool
    out_of_scope_current_query: bool
    parts_query: bool
    manager_query: bool
    nonsense_input: bool
    abuse_message: bool
    pricing_request: bool
    catalog_list_request: bool
    feature_comparison: bool
    refund_policy: bool
    post_payment_access: bool
    multi_user_access: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PlannerDecision:
    current_focus: str = "unknown"
    planned_action: str = ""
    secondary_actions: list[str] = field(default_factory=list)
    response_plan: list[str] = field(default_factory=list)
    clarify_required: bool = False
    flow_name: str = "none"
    flow_step: str = "idle"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class StateProjection:
    state_before: dict[str, object]
    state_after: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {"state_before": self.state_before, "state_after": self.state_after}
