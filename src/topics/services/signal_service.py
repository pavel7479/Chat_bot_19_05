from __future__ import annotations

from typing import Any


class SignalService:
    """Service wrapper for semantic signal extraction."""

    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor

    def extract(self, context: Any) -> set[str]:
        return self._extractor.extract(context)

