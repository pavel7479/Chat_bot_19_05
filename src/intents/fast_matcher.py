from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FastMatchResult:
    intent: str
    confidence: float
    slots: dict[str, object]
    evidence: str


class FastMatcher:
    def __init__(self, rules: dict[str, Any]) -> None:
        intents_raw = rules.get("intents", []) if isinstance(rules, dict) else []
        self._intents: list[tuple[str, list[re.Pattern[str]], dict[str, object], float, int]] = []
        for item in intents_raw:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip()
            if not intent:
                continue
            patterns_raw = item.get("patterns", [])
            patterns: list[re.Pattern[str]] = []
            if isinstance(patterns_raw, list):
                for raw in patterns_raw:
                    expr = str(raw).strip()
                    if not expr:
                        continue
                    patterns.append(re.compile(expr, flags=re.IGNORECASE))
            if not patterns:
                continue
            slots = item.get("slots", {}) if isinstance(item.get("slots", {}), dict) else {}
            confidence = float(item.get("confidence", 0.95))
            priority = int(item.get("priority", 0))
            self._intents.append((intent, patterns, slots, confidence, priority))

    def match(self, query: str) -> FastMatchResult | None:
        normalized = query.strip()
        if not normalized:
            return None
        best: tuple[FastMatchResult, int, int] | None = None
        for intent, patterns, slots, confidence, priority in self._intents:
            for pattern in patterns:
                matched = pattern.search(normalized)
                if matched:
                    result = FastMatchResult(
                        intent=intent,
                        confidence=max(0.0, min(1.0, confidence)),
                        slots=dict(slots),
                        evidence=pattern.pattern,
                    )
                    span_len = max(0, matched.end() - matched.start())
                    if best is None:
                        best = (result, priority, span_len)
                        continue
                    best_result, best_priority, best_span = best
                    if priority > best_priority:
                        best = (result, priority, span_len)
                        continue
                    if priority == best_priority and result.confidence > best_result.confidence:
                        best = (result, priority, span_len)
                        continue
                    if (
                        priority == best_priority
                        and result.confidence == best_result.confidence
                        and span_len > best_span
                    ):
                        best = (result, priority, span_len)
        return best[0] if best is not None else None
