from __future__ import annotations

from typing import Any

from src.topics.dto import PlannerDecision, TurnSignals


class PlannerDecisionService:
    """Builds typed planner decision DTO from planner output."""

    def __init__(self, planning_service: Any) -> None:
        self._planning_service = planning_service

    def decide(
        self,
        topic_ids: list[str],
        turn_signals: TurnSignals,
        state_snapshot: dict[str, object],
        user_query: str,
    ) -> PlannerDecision:
        plan = self._planning_service.plan(
            topic_ids=topic_ids,
            turn_analysis=turn_signals.as_dict(),
            state_snapshot=state_snapshot,
            user_query=user_query,
        )
        return PlannerDecision(
            current_focus=str(plan.get("current_focus", "unknown")),
            planned_action=str(plan.get("planned_action", "")),
            secondary_actions=[str(item) for item in plan.get("secondary_actions", []) if str(item).strip()],
            response_plan=[str(item) for item in plan.get("response_plan", []) if str(item).strip()],
            clarify_required=bool(plan.get("clarify_required", False)),
            flow_name=str(plan.get("flow_name", "none")),
            flow_step=str(plan.get("flow_step", "idle")),
        )
