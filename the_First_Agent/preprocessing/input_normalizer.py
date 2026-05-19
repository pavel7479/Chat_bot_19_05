from __future__ import annotations

import re


class InputNormalizationService:
    def normalize(self, query: str) -> dict[str, object]:
        original = query or ""
        stripped = original.strip()
        collapsed = re.sub(r"\s+", " ", stripped)
        lowered = collapsed.lower()
        return {
            "original_query": original,
            "normalized_query": collapsed,
            "lowered_query": lowered,
            "changed": collapsed != original,
        }
