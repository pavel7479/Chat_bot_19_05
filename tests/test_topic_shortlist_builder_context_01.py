from __future__ import annotations

import unittest

from src.core.models import ContextSignals, SessionState
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.config.resource_paths import SEMANTIC_INTENTS_PATH


class TopicShortlistBuilderContext01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.builder = TopicShortlistBuilder(cls.topic_catalog.topics, top_k=8)

    def test_meaning_keeps_legal_entity_topic_for_short_reply(self) -> None:
        shortlist = self.builder.build_shortlist(
            "являюсь",
            history_text="бот: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?",
            session_state=SessionState(),
            context_signals=ContextSignals(
                user_query="являюсь",
                meaning="Клиент подтверждает статус представителя автобизнеса для демо-доступа.",
                gist="Обсуждается демо-доступ.",
            ),
        )
        topic_ids = [item.topic_id for item in shortlist]
        self.assertIn("legal_entity_purchase_flow", topic_ids)
        self.assertIn("demo_access", topic_ids)
        trace = self.builder.get_last_semantic_routing_trace()
        self.assertTrue(trace["meaning_used"])

    def test_brand_typo_boost_keeps_specific_brand_check_in_shortlist(self) -> None:
        shortlist = self.builder.build_shortlist(
            "а Mercedec и Wolksvagen есть?",
            history_text="клиент: Какие бренды есть?\nбот: Система охватывает все мировые бренды.",
            session_state=SessionState(),
            context_signals=ContextSignals(
                user_query="а Mercedec и Wolksvagen есть?",
                meaning="Клиент уточняет наличие конкретных брендов в каталоге.",
                gist="Уточнение по брендам.",
            ),
        )
        topic_ids = [item.topic_id for item in shortlist]
        self.assertIn("specific_brand_check", topic_ids)
        trace = self.builder.get_last_semantic_routing_trace()
        self.assertTrue(trace["brand_alias_hits"] or trace["brand_fuzzy_hits"])

    def test_zero_score_fallback_is_not_empty(self) -> None:
        shortlist = self.builder.build_shortlist(
            "ываыва абракадабра",
            history_text="",
            session_state=SessionState(),
            context_signals=ContextSignals(
                user_query="ываыва абракадабра",
                meaning="",
                gist="",
                fallback_used=True,
            ),
        )
        self.assertGreater(len(shortlist), 0)
        trace = self.builder.get_last_semantic_routing_trace()
        self.assertTrue(trace["zero_score_fallback_used"])

    def test_fallback_disables_meaning_scoring(self) -> None:
        shortlist = self.builder.build_shortlist(
            "....548",
            history_text="бот: Вы представитель автобизнеса?",
            session_state=SessionState(),
            context_signals=ContextSignals(
                user_query="....548",
                meaning="Клиент подтверждает статус представителя автобизнеса.",
                gist="Обсуждается демо-доступ.",
                fallback_used=True,
            ),
        )
        trace = self.builder.get_last_semantic_routing_trace()
        self.assertTrue(trace["fallback_used"])
        self.assertFalse(trace["meaning_used"])
        self.assertIn("nonsense_input", [item.topic_id for item in shortlist])


if __name__ == "__main__":
    unittest.main()
