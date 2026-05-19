from __future__ import annotations

from typing import Any


class ContextService:
    """Service wrapper for prompt context extraction."""

    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor

    def extract(self, prompt: str) -> Any:
        return self._extractor.extract(prompt)

