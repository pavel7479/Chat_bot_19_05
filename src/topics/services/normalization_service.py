from __future__ import annotations

from typing import Any


class TopicNormalizationService:
    """Service wrapper for topic normalization policy."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def resolve_locked_topics(self, context: Any) -> set[str]:
        return self._policy.resolve_locked_topics(context)

    def normalize(
        self,
        llm_topics: list[str],
        context: Any,
        state: Any,
        signals: set[str],
        locked_topics: set[str],
    ) -> tuple[list[str], list[dict[str, object]]]:
        return self._policy.normalize(
            llm_topics=llm_topics,
            context=context,
            state=state,
            signals=signals,
            locked_topics=locked_topics,
        )

