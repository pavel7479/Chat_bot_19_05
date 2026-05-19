from __future__ import annotations

import unittest

from src.core.models import SessionState
from the_First_Agent.config.resource_paths import CONTEXT_SIGNAL_RULES_PATH
from the_First_Agent.context.context_signal_extractor import ContextSignalExtractor


class ContextSignalExtractor01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = ContextSignalExtractor(CONTEXT_SIGNAL_RULES_PATH)

    def test_confirmed_legal_status_and_demo_context_are_split(self) -> None:
        signals = self.extractor.extract(
            user_query="являюсь",
            meaning="Клиент подтверждает, что является представителем автобизнеса.",
            gist="Обсуждается получение тестового доступа.",
            history_text="бот: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?",
            session_state=SessionState(),
            fallback_used=False,
        )
        self.assertIn("confirmed_legal_status", signals.semantic_flags)
        self.assertIn("legal_entity_purchase_flow", signals.semantic_boost_topics)
        self.assertIn("demo_access", signals.continuity_topics)

    def test_switch_to_tis_adds_penalty_for_epc(self) -> None:
        signals = self.extractor.extract(
            user_query="тогда только TIS",
            meaning="Клиент переключается на отдельный продукт TIS.",
            gist="Обсуждается предыдущая тема EPC.",
            history_text="бот: EPC продается полным пакетом",
            session_state=SessionState(last_topic_ids=["epc_tariffs"]),
            fallback_used=False,
        )
        self.assertIn("switch_to_tis", signals.semantic_flags)
        self.assertIn("tis_tariffs", signals.semantic_boost_topics)
        self.assertIn("epc_tariffs", signals.semantic_penalty_topics)

    def test_fallback_disables_semantic_influence(self) -> None:
        signals = self.extractor.extract(
            user_query="....548",
            meaning="Последняя реплика клиента: ....548",
            gist="Не удалось надежно определить суть диалога.",
            history_text="бот: Вы представитель автобизнеса?",
            session_state=SessionState(last_topic_ids=["demo_access"]),
            fallback_used=True,
        )
        self.assertTrue(signals.fallback_used)
        self.assertEqual(signals.semantic_flags, set())
        self.assertEqual(signals.semantic_boost_topics, set())
        self.assertEqual(signals.semantic_penalty_topics, set())
        self.assertEqual(signals.continuity_topics, set())


if __name__ == "__main__":
    unittest.main()
