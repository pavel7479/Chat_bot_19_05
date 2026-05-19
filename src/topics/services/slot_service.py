from __future__ import annotations

from typing import Any


class SlotService:
    """Service wrapper for semantic slot extraction."""

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver

    def resolve(self, context: Any) -> dict[str, bool]:
        return self._resolver.resolve(context)

