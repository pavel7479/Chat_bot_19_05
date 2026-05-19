from __future__ import annotations

from typing import Any


class DialogueStateService:
    """Service wrapper for dialogue state resolve/transition."""

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver

    def resolve(self, context: Any, session_state: Any = None) -> Any:
        return self._resolver.resolve(context, session_state=session_state)

    def transition(
        self,
        resolved_state: Any,
        topics: list[str],
        context: Any,
        state_update_raw: object = None,
    ) -> Any:
        return self._resolver.next_state(resolved_state, topics, context, state_update_raw)

