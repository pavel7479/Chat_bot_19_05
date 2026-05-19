from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.models import ChatMessage, SessionState
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.config.resource_paths import SEMANTIC_INTENTS_PATH
from the_First_Agent.context.history_formatter import format_history


class TopicShortlistBuilderStage401Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.builder = TopicShortlistBuilder(cls.topic_catalog.topics, top_k=8)

    @staticmethod
    def _history_text(history: list[tuple[str, str]]) -> str:
        messages = [ChatMessage(role=role, text=text) for role, text in history]
        return format_history(messages)

    def test_purchase_ready_not_boosted_by_payment_without_details_phrase(self) -> None:
        shortlist = self.builder.build_shortlist("Оплачу, но ИНН не дам", history_text="", session_state=SessionState())
        topic_ids = [item.topic_id for item in shortlist]
        self.assertIn("payment_without_details", topic_ids)
        if "purchase_ready" in topic_ids:
            self.assertLess(topic_ids.index("payment_without_details"), topic_ids.index("purchase_ready"))

    def test_payment_without_details_patterns_cover_requisites_refusal(self) -> None:
        shortlist = self.builder.build_shortlist("реквизиты не дам", history_text="", session_state=SessionState())
        self.assertIn("payment_without_details", [item.topic_id for item in shortlist])

    def test_assistant_hints_do_not_affect_regular_pricing_query(self) -> None:
        history = self._history_text(
            [
                ("user", "Хочу купить"),
                ("assistant", "Вы юридическое лицо? Напишите ИНН и реквизиты."),
            ]
        )
        shortlist = self.builder.build_shortlist("сколько стоит подписка", history_text=history, session_state=SessionState())
        topic_ids = [item.topic_id for item in shortlist]
        self.assertIn("epc_tariffs", topic_ids)
        self.assertIn("tis_tariffs", topic_ids)
        self.assertNotIn("legal_entity_purchase_flow", topic_ids[:2])

    def test_general_pricing_ensures_both_pricing_topics(self) -> None:
        shortlist = self.builder.build_shortlist("сколько стоит подписка", history_text="", session_state=SessionState())
        topic_ids = [item.topic_id for item in shortlist]
        self.assertIn("epc_tariffs", topic_ids)
        self.assertIn("tis_tariffs", topic_ids)


if __name__ == "__main__":
    unittest.main()
