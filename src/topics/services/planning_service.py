from __future__ import annotations

from typing import Any


class PlanningService:
    """Service wrapper for planner action resolver."""

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver

    def plan(
        self,
        topic_ids: list[str],
        turn_analysis: dict[str, object],
        state_snapshot: dict[str, object],
        user_query: str,
    ) -> dict[str, object]:
        return self._resolver.plan(
            topic_ids=topic_ids,
            turn_analysis=turn_analysis,
            state_snapshot=state_snapshot,
            user_query=user_query,
        )

