from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class CarryoverContext:
    last_question_type: str
    short_reply_polarity: str
    last_assistant_message: str
    history_lines: list[str]


class ContextCarryoverResolver:
    """Resolves short contextual follow-ups into stable intent carryover."""

    def __init__(self, topic_ids: Iterable[str]) -> None:
        self._topic_ids = set(str(item).strip() for item in topic_ids if str(item).strip())

    def resolve(self, routed_topics: list[str], context: CarryoverContext) -> list[str]:
        if not routed_topics:
            return routed_topics

        polarity = str(context.short_reply_polarity or "unknown")
        question_type = str(context.last_question_type or "unknown")
        last_assistant = str(context.last_assistant_message or "").lower()
        history_tail = " ".join(str(line).lower() for line in (context.history_lines or [])[-3:])

        if question_type == "unknown":
            source_text = f"{last_assistant} {history_tail}".strip()
            if "счет" in source_text or "счёт" in source_text or "инн" in source_text:
                question_type = "invoice_confirmation"
            elif "юр" in source_text or "юрид" in source_text or "ип" in source_text:
                question_type = "legal_status_check"
            elif "оформ" in source_text or "покуп" in source_text:
                question_type = "purchase_confirmation"

        if polarity == "yes":
            override_map = {
                "legal_status_check": "legal_entity_purchase_flow",
                "demo_legal_check": "demo_access",
                "invoice_confirmation": "legal_entity_purchase_flow",
                "purchase_confirmation": "purchase_ready",
            }
            topic = override_map.get(question_type)
            if topic in self._topic_ids:
                return self._merge_priority_topic(topic, routed_topics)
            return routed_topics

        if polarity == "no":
            override_map = {
                "legal_status_check": "physical_person_purchase",
                "demo_legal_check": "physical_person_purchase",
                "purchase_confirmation": "purchase_ready",
            }
            topic = override_map.get(question_type)
            if topic in self._topic_ids:
                return self._merge_priority_topic(topic, routed_topics)
            return routed_topics

        return routed_topics

    def _merge_priority_topic(self, topic: str, routed_topics: list[str]) -> list[str]:
        # Keep contextual intent as primary for short/confirm turns, but preserve a second topic when present.
        out = [topic]
        for item in routed_topics:
            if item != topic and item not in out:
                out.append(item)
            if len(out) >= 2:
                break
        return out
