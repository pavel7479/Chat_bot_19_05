from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.agents.intent_agent import IntentAgent
from src.config.loader import ConfigLoader
from src.app.state_update_service import StateUpdateService
from src.core.models import SessionState, TopicClassificationResult
from src.prompting.prompt_manager import PromptManager
from the_First_Agent.Agent_Zero.models import ContextUnderstandingResult
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.config.resource_paths import CONTEXT_SIGNAL_RULES_PATH, SEMANTIC_INTENTS_PATH, TOPIC_CLASSIFIER_PROMPT_PATH
from the_First_Agent.context.context_signal_extractor import ContextSignalExtractor
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder


class _FakeContextUnderstandingAgent:
    def __init__(self, result: ContextUnderstandingResult) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def understand(self, dialog_text: str, user_query: str) -> ContextUnderstandingResult:
        self.calls.append({"dialog_text": dialog_text, "user_query": user_query})
        return self.result


class _FakeTopicClassifier:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls: list[dict[str, object]] = []

    def classify(
        self,
        prompt: str,
        *,
        dialog_text: str,
        user_query: str,
        context_signals=None,
        session_state: SessionState | None = None,
    ) -> TopicClassificationResult:
        self.prompts.append(prompt)
        self.calls.append(
            {
                "dialog_text": dialog_text,
                "user_query": user_query,
                "context_signals": context_signals,
            }
        )
        return TopicClassificationResult(
            topic_ids=["legal_entity_purchase_flow"],
            confidence=0.91,
            reason="Клиент подтвердил юридический статус.",
            diagnostics={
                "first_agent_output": {
                    "prompt": prompt,
                    "raw_llm_response": "{}",
                    "parsed_json": {},
                    "topic_ids": ["legal_entity_purchase_flow"],
                    "intent_scores": [],
                    "intent_details": [],
                    "intent_reasons": [],
                    "confidence": 0.91,
                    "reason": "Клиент подтвердил юридический статус.",
                    "state_snapshot": {},
                    "llm_attempts": [],
                    "schema_retry_used": False,
                },
                "first_agent_trace": {"pipeline_steps": []},
                "state_trace": [],
            },
        )


class IntentAgentContextUnderstanding01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            cls.config.paths.answer_generator_prompt,
        )
        cls.topic_prompt_sections_builder = TopicPromptSectionsBuilder()

    def _build_agent(self, context_result: ContextUnderstandingResult) -> tuple[IntentAgent, _FakeContextUnderstandingAgent, _FakeTopicClassifier]:
        fake_context_agent = _FakeContextUnderstandingAgent(context_result)
        fake_classifier = _FakeTopicClassifier()
        agent = IntentAgent(
            topic_catalog=self.topic_catalog,
            topic_shortlist_builder=TopicShortlistBuilder(self.topic_catalog.topics, top_k=8),
            topic_classifier=fake_classifier,
            prompt_manager=self.prompt_manager,
            topic_prompt_sections_builder=self.topic_prompt_sections_builder,
            context_understanding_agent=fake_context_agent,
            context_signal_extractor=ContextSignalExtractor(CONTEXT_SIGNAL_RULES_PATH),
        )
        return agent, fake_context_agent, fake_classifier

    def test_context_understanding_is_called_and_embedded_into_prompt(self) -> None:
        agent, fake_context, fake_classifier = self._build_agent(
            ContextUnderstandingResult(
                gist="Клиент интересуется демо-доступом, бот уточняет статус.",
                meaning="Клиент подтвердил, что является представителем автобизнеса.",
                raw_response='{"gist":"x","meaning":"y"}',
                parsed_json={"gist": "x", "meaning": "y"},
            )
        )
        prompt, result = agent.classify(
            history_text="user: Можно демо?\nassistant: Вы представитель автобизнеса?",
            user_query="Да, являюсь",
            session_state=SessionState(),
        )
        self.assertEqual(len(fake_context.calls), 1)
        self.assertIn("Промежуточное понимание диалога:", prompt)
        self.assertIn("- Суть диалога: Клиент интересуется демо-доступом, бот уточняет статус.", prompt)
        self.assertIn("- Смысл последней реплики: Клиент подтвердил, что является представителем автобизнеса.", prompt)
        self.assertIn("Клиент подтвердил, что является представителем автобизнеса.", prompt)
        self.assertNotIn('{"gist"', prompt)
        self.assertEqual(fake_classifier.prompts[-1], prompt)
        self.assertEqual(fake_classifier.calls[-1]["user_query"], "Да, являюсь")
        context_diag = result.diagnostics.get("context_understanding", {})
        self.assertEqual(context_diag.get("gist"), "Клиент интересуется демо-доступом, бот уточняет статус.")
        self.assertEqual(context_diag.get("meaning"), "Клиент подтвердил, что является представителем автобизнеса.")
        self.assertEqual(result.state_snapshot.get("last_context_gist"), "Клиент интересуется демо-доступом, бот уточняет статус.")
        self.assertEqual(result.state_snapshot.get("last_context_meaning"), "Клиент подтвердил, что является представителем автобизнеса.")
        self.assertFalse(result.state_snapshot.get("last_context_fallback_used"))

    def test_context_understanding_trace_is_honest(self) -> None:
        agent, _, _ = self._build_agent(
            ContextUnderstandingResult(
                gist="Не удалось надежно определить суть диалога.",
                meaning="Последняя реплика клиента: ....548",
                raw_response="",
                parsed_json={},
                fallback_used=True,
                fallback_reason="json_object_not_found",
            )
        )
        _, result = agent.classify(
            history_text="assistant: Вы представитель автобизнеса?",
            user_query="....548",
            session_state=SessionState(),
        )
        trace = result.diagnostics.get("first_agent_trace", {}).get("pipeline_steps", [])
        self.assertTrue(trace)
        self.assertEqual(trace[0].get("step"), "context_understanding")
        self.assertEqual(trace[0].get("status"), "fallback")
        self.assertEqual(trace[0].get("after", {}).get("meaning"), "Последняя реплика клиента: ....548")
        self.assertEqual(result.diagnostics.get("context_understanding", {}).get("fallback_reason"), "json_object_not_found")
        self.assertTrue(result.state_snapshot.get("last_context_fallback_used"))
        self.assertEqual(result.state_snapshot.get("last_context_fallback_reason"), "json_object_not_found")

    def test_fallback_is_safe_when_context_agent_is_missing(self) -> None:
        fake_classifier = _FakeTopicClassifier()
        agent = IntentAgent(
            topic_catalog=self.topic_catalog,
            topic_shortlist_builder=TopicShortlistBuilder(self.topic_catalog.topics, top_k=8),
            topic_classifier=fake_classifier,
            prompt_manager=self.prompt_manager,
            topic_prompt_sections_builder=self.topic_prompt_sections_builder,
            context_understanding_agent=None,
            context_signal_extractor=ContextSignalExtractor(CONTEXT_SIGNAL_RULES_PATH),
        )
        prompt, result = agent.classify(
            history_text="user: Хочу купить доступ",
            user_query="Да, являюсь",
            session_state=SessionState(),
        )
        self.assertIn("Промежуточное понимание диалога:", prompt)
        context_diag = result.diagnostics.get("context_understanding", {})
        self.assertTrue(context_diag.get("fallback_used"))
        self.assertEqual(context_diag.get("fallback_reason"), "context_understanding_agent_not_configured")
        self.assertIn("Последняя реплика клиента: Да, являюсь", prompt)
        self.assertTrue(result.state_snapshot.get("last_context_fallback_used"))

    def test_context_understanding_state_is_persisted_by_state_update_service(self) -> None:
        agent, _, _ = self._build_agent(
            ContextUnderstandingResult(
                gist="Клиент интересуется демо-доступом.",
                meaning="Клиент подтвердил юридический статус.",
                raw_response='{"gist":"x","meaning":"y"}',
                parsed_json={"gist": "x", "meaning": "y"},
            )
        )
        _, result = agent.classify(
            history_text="user: Можно демо?\nassistant: Вы юрлицо?",
            user_query="Да, являюсь",
            session_state=SessionState(),
        )
        updated_state = StateUpdateService().apply_after_classification(
            base_state=SessionState(),
            topic_result=result,
        )
        self.assertEqual(updated_state.last_context_gist, "Клиент интересуется демо-доступом.")
        self.assertEqual(updated_state.last_context_meaning, "Клиент подтвердил юридический статус.")
        self.assertFalse(updated_state.last_context_fallback_used)

    def test_prompt_places_context_block_after_last_phrase(self) -> None:
        agent, _, _ = self._build_agent(
            ContextUnderstandingResult(
                gist="Клиент интересуется демо-доступом.",
                meaning="Клиент подтвердил юридический статус.",
                raw_response='{"gist":"x","meaning":"y"}',
                parsed_json={"gist": "x", "meaning": "y"},
            )
        )
        prompt, _ = agent.classify(
            history_text="user: Можно демо?\nassistant: Вы юрлицо?",
            user_query="Да, являюсь",
            session_state=SessionState(),
        )
        dialog_pos = prompt.rfind("Диалог:\nклиент: Можно демо?\nбот: Вы юрлицо?\nклиент: Да, являюсь")
        last_phrase_pos = prompt.find("Последняя реплика клиента:\nДа, являюсь", dialog_pos)
        context_pos = prompt.find("Промежуточное понимание диалога:", last_phrase_pos)
        allowed_pos = prompt.index("Разрешённые intent_id:")
        self.assertLess(dialog_pos, last_phrase_pos)
        self.assertLess(last_phrase_pos, context_pos)
        self.assertLess(context_pos, allowed_pos)


if __name__ == "__main__":
    unittest.main()
