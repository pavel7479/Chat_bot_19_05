from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.agents.intent_agent import IntentAgent
from src.agents.response_planner import ResponsePlanner
from src.config.loader import ConfigLoader
from src.core.interfaces import LLMProvider
from src.core.models import SemanticFrame, SessionState, TopicClassificationResult
from src.main import build_app
from src.prompting.prompt_manager import PromptManager
from the_First_Agent.Agent_Zero.context_understanding_agent import ContextUnderstandingAgent
from the_First_Agent.Agent_Zero.models import ContextUnderstandingResult
from the_First_Agent.Agent_Zero.parser import ContextUnderstandingParser
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.config.resource_paths import (
    CONTEXT_SIGNAL_RULES_PATH,
    SEMANTIC_INTENTS_PATH,
    TOPIC_CLASSIFIER_PROMPT_PATH,
)
from the_First_Agent.context.context_signal_extractor import ContextSignalExtractor
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder


class _StaticJsonLLM(LLMProvider):
    def __init__(self, payload: dict[str, object] | str) -> None:
        self._payload = payload
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        return ""

    def generate_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload, ensure_ascii=False)


class _FakeContextUnderstandingAgent:
    def __init__(self, result: ContextUnderstandingResult) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def understand(self, dialog_text: str, user_query: str) -> ContextUnderstandingResult:
        self.calls.append({"dialog_text": dialog_text, "user_query": user_query})
        return self.result


class _MappedContextUnderstandingAgent:
    def __init__(self, *, explicit_semantic_frame: bool) -> None:
        self._explicit_semantic_frame = explicit_semantic_frame

    def understand(self, dialog_text: str, user_query: str) -> ContextUnderstandingResult:
        scenarios = {
            "Какой каталог мне нужен для Mercedes?": (
                "Клиент выбирает подходящий каталог для Mercedes.",
                "Клиент хочет понять, какой каталог подходит для его задачи по Mercedes.",
                "product_choice",
                "ask_product_recommendation",
            ),
            "Оплатил, но доступа нет": (
                "Клиент сообщает о проблеме после оплаты.",
                "Клиент просит помочь с отсутствием доступа после оплаты.",
                "support",
                "ask_support",
            ),
            "Как купить подписку?": (
                "Клиент хочет узнать порядок покупки подписки.",
                "Клиент спрашивает, как оформить покупку доступа.",
                "purchase",
                "ask_purchase_steps",
            ),
        }
        gist, meaning, conversation_mode, user_goal = scenarios[user_query]
        semantic_frame = None
        if self._explicit_semantic_frame:
            semantic_frame = SemanticFrame(
                conversation_mode=conversation_mode,
                user_goal=user_goal,
                is_followup=False,
                is_topic_switch=False,
                language="ru",
                confidence=0.9,
                gist=gist,
                meaning=meaning,
            )
        return ContextUnderstandingResult(
            gist=gist,
            meaning=meaning,
            raw_response='{"gist":"x","meaning":"y"}',
            parsed_json={"gist": gist, "meaning": meaning},
            semantic_frame=semantic_frame,
        )


class _QueryMappedTopicClassifier:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls: list[dict[str, object]] = []
        self._mapping: dict[str, tuple[list[str], float, str]] = {
            "Какой каталог мне нужен для Mercedes?": (
                ["specific_brand_check"],
                0.86,
                "Запрос про каталог для конкретного бренда.",
            ),
            "Оплатил, но доступа нет": (
                ["post_payment_no_access"],
                0.93,
                "Проблема с доступом после оплаты.",
            ),
            "Как купить подписку?": (
                ["purchase_ready"],
                0.91,
                "Запрос о порядке оформления подписки.",
            ),
        }

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
        topic_ids, confidence, reason = self._mapping.get(
            user_query,
            (["company_services_info"], 0.7, "stub"),
        )
        return TopicClassificationResult(
            topic_ids=topic_ids,
            confidence=confidence,
            reason=reason,
            diagnostics={
                "first_agent_output": {"state_snapshot": {}},
                "first_agent_trace": {"pipeline_steps": []},
                "state_trace": [],
            },
        )


class _StaticAnswerLLM(LLMProvider):
    def generate_text(self, prompt: str) -> str:
        return ""

    def generate_json(self, prompt: str) -> str:
        return json.dumps({"answer_text": "Smoke answer from fake llm."}, ensure_ascii=False)


class Stage01SemanticFrameLightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.prompt_path = cls.project_root / cls.config.paths.context_understanding_prompt
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            cls.config.paths.answer_generator_prompt,
        )
        cls.topic_prompt_sections_builder = TopicPromptSectionsBuilder()
        cls.response_planner = ResponsePlanner(
            cls.project_root / "src/config/response_fact_map.yaml",
            cls.project_root / cls.config.paths.brands_file,
            facts_file_path=cls.project_root / "src/config/facts.yaml",
        )

    def _build_intent_agent(
        self,
        *,
        context_result: ContextUnderstandingResult,
    ) -> tuple[IntentAgent, _FakeContextUnderstandingAgent, _QueryMappedTopicClassifier]:
        fake_context_agent = _FakeContextUnderstandingAgent(context_result)
        fake_classifier = _QueryMappedTopicClassifier()
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

    def test_agent_zero_accepts_full_semantic_frame_contract(self) -> None:
        llm = _StaticJsonLLM(
            {
                "gist": "Клиент выбирает подходящий каталог для Mercedes.",
                "meaning": "Клиент хочет понять, какой каталог подходит для его задачи по Mercedes.",
                "turn_type": "product_choice",
                "turn_subtype": "brand_recommendation",
                "confidence": 0.88,
                "semantic_flags": ["brand_present"],
                "semantic_frame": {
                    "conversation_mode": "product_choice",
                    "user_goal": "ask_product_recommendation",
                    "is_followup": False,
                    "is_topic_switch": False,
                    "language": "ru",
                    "confidence": 0.88,
                    "gist": "Клиент выбирает подходящий каталог для Mercedes.",
                    "meaning": "Клиент хочет понять, какой каталог подходит для его задачи по Mercedes.",
                },
            }
        )
        agent = ContextUnderstandingAgent(llm, self.prompt_path)

        result = agent.understand(
            dialog_text="бот: Опишите задачу по каталогу.",
            user_query="Какой каталог мне нужен для Mercedes?",
        )

        self.assertFalse(result.fallback_used)
        self.assertIsNotNone(result.semantic_frame)
        self.assertEqual(result.turn_type, "product_choice")
        self.assertEqual(result.semantic_frame.conversation_mode, "product_choice")
        self.assertEqual(result.semantic_frame.user_goal, "ask_product_recommendation")
        self.assertEqual(result.semantic_frame.language, "ru")

    def test_parser_builds_safe_semantic_frame_when_llm_returns_legacy_json(self) -> None:
        parser = ContextUnderstandingParser()

        result = parser.parse(
            raw_response='{"gist":"Клиент сообщает о проблеме после оплаты.","meaning":"Клиент просит помочь с отсутствием доступа после оплаты."}',
            user_query="Оплатил, но доступа нет",
        )

        self.assertFalse(result.fallback_used)
        self.assertIsNotNone(result.semantic_frame)
        self.assertEqual(result.semantic_frame.conversation_mode, "unknown")
        self.assertEqual(result.semantic_frame.user_goal, "unknown")
        self.assertEqual(result.semantic_frame.language, "ru")

    def test_parser_invalid_semantic_frame_does_not_break_pipeline(self) -> None:
        parser = ContextUnderstandingParser()

        result = parser.parse(
            raw_response=json.dumps(
                {
                    "gist": "Клиент хочет узнать порядок покупки подписки.",
                    "meaning": "Клиент спрашивает, как оформить покупку доступа.",
                    "semantic_frame": {
                        "conversation_mode": "purchase",
                        "user_goal": "ask_purchase_steps",
                        "is_followup": False,
                        "is_topic_switch": False,
                        "language": "ru",
                        "confidence": 0.92,
                        "gist": "Клиент хочет узнать порядок покупки подписки.",
                        "meaning": "Клиент спрашивает, как оформить покупку доступа.",
                        "unexpected": "boom",
                    },
                },
                ensure_ascii=False,
            ),
            user_query="Как купить подписку?",
        )

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "schema_validation_failed")
        self.assertIsNotNone(result.semantic_frame)
        self.assertEqual(result.semantic_frame.conversation_mode, "unknown")
        self.assertEqual(result.semantic_frame.user_goal, "unknown")

    def test_intent_agent_writes_provided_semantic_frame_to_diagnostics_and_state(self) -> None:
        user_query = "Какой каталог мне нужен для Mercedes?"
        history_text = "бот: Уточню, какая задача вам нужна."
        context_result = ContextUnderstandingResult(
            gist="Клиент выбирает подходящий каталог для Mercedes.",
            meaning="Клиент хочет понять, какой каталог лучше подходит для его задачи по Mercedes.",
            raw_response='{"gist":"x","meaning":"y","semantic_frame":{}}',
            parsed_json={"gist": "x", "meaning": "y"},
            semantic_frame=SemanticFrame(
                conversation_mode="product_choice",
                user_goal="ask_product_recommendation",
                is_followup=False,
                is_topic_switch=False,
                language="ru",
                confidence=0.84,
                gist="Клиент выбирает подходящий каталог для Mercedes.",
                meaning="Клиент хочет понять, какой каталог лучше подходит для его задачи по Mercedes.",
            ),
        )
        agent, _, _ = self._build_intent_agent(context_result=context_result)

        _, result = agent.classify(
            history_text=history_text,
            user_query=user_query,
            session_state=SessionState(),
        )

        semantic_frame = result.diagnostics.get("semantic_frame", {})
        self.assertEqual(semantic_frame.get("conversation_mode"), "product_choice")
        self.assertEqual(semantic_frame.get("user_goal"), "ask_product_recommendation")
        self.assertEqual(result.state_snapshot.get("conversation_mode"), "product_choice")
        self.assertEqual(result.state_snapshot.get("user_goal"), "ask_product_recommendation")
        self.assertEqual(
            result.state_snapshot.get("last_semantic_frame", {}).get("conversation_mode"),
            "product_choice",
        )

    def test_semantic_frame_does_not_change_topic_or_action_for_stage01_scenarios(self) -> None:
        scenarios = [
            {
                "user_query": "Какой каталог мне нужен для Mercedes?",
                "history_text": "бот: Опишите задачу по Mercedes.",
                "gist": "Клиент выбирает подходящий каталог для Mercedes.",
                "meaning": "Клиент хочет понять, какой каталог подходит для его задачи по Mercedes.",
                "expected_topic": ["specific_brand_check"],
                "expected_action": "unknown_brand_unavailable",
                "conversation_mode": "product_choice",
                "user_goal": "ask_product_recommendation",
            },
            {
                "user_query": "Оплатил, но доступа нет",
                "history_text": "бот: Опишите проблему с доступом.",
                "gist": "Клиент сообщает о проблеме после оплаты.",
                "meaning": "Клиент просит помочь с отсутствием доступа после оплаты.",
                "expected_topic": ["post_payment_no_access"],
                "expected_action": "post_payment_no_access_handoff",
                "conversation_mode": "support",
                "user_goal": "ask_support",
            },
            {
                "user_query": "Как купить подписку?",
                "history_text": "бот: Подскажу по оформлению доступа.",
                "gist": "Клиент хочет узнать порядок покупки подписки.",
                "meaning": "Клиент спрашивает, как оформить покупку доступа.",
                "expected_topic": ["purchase_ready"],
                "expected_action": "ask_legal_status",
                "conversation_mode": "purchase",
                "user_goal": "ask_purchase_steps",
            },
        ]

        for scenario in scenarios:
            with self.subTest(user_query=scenario["user_query"]):
                legacy_result = ContextUnderstandingResult(
                    gist=scenario["gist"],
                    meaning=scenario["meaning"],
                    raw_response='{"gist":"legacy","meaning":"legacy"}',
                    parsed_json={"gist": scenario["gist"], "meaning": scenario["meaning"]},
                )
                explicit_result = ContextUnderstandingResult(
                    gist=scenario["gist"],
                    meaning=scenario["meaning"],
                    raw_response='{"gist":"full","meaning":"full","semantic_frame":{}}',
                    parsed_json={"gist": scenario["gist"], "meaning": scenario["meaning"]},
                    semantic_frame=SemanticFrame(
                        conversation_mode=scenario["conversation_mode"],
                        user_goal=scenario["user_goal"],
                        is_followup=False,
                        is_topic_switch=False,
                        language="ru",
                        confidence=0.9,
                        gist=scenario["gist"],
                        meaning=scenario["meaning"],
                    ),
                )

                legacy_agent, _, _ = self._build_intent_agent(context_result=legacy_result)
                semantic_agent, _, _ = self._build_intent_agent(context_result=explicit_result)

                _, legacy_topic = legacy_agent.classify(
                    history_text=scenario["history_text"],
                    user_query=scenario["user_query"],
                    session_state=SessionState(),
                )
                _, semantic_topic = semantic_agent.classify(
                    history_text=scenario["history_text"],
                    user_query=scenario["user_query"],
                    session_state=SessionState(),
                )
                legacy_plan = self.response_planner.plan(
                    topic_result=legacy_topic,
                    user_query=scenario["user_query"],
                    history_text=scenario["history_text"],
                )
                semantic_plan = self.response_planner.plan(
                    topic_result=semantic_topic,
                    user_query=scenario["user_query"],
                    history_text=scenario["history_text"],
                )

                self.assertEqual(legacy_topic.topic_ids, scenario["expected_topic"])
                self.assertEqual(semantic_topic.topic_ids, scenario["expected_topic"])
                self.assertEqual(legacy_plan.primary_action, scenario["expected_action"])
                self.assertEqual(semantic_plan.primary_action, scenario["expected_action"])
                self.assertEqual(legacy_topic.topic_ids, semantic_topic.topic_ids)
                self.assertEqual(legacy_plan.primary_action, semantic_plan.primary_action)

    def test_chatbot_orchestrator_smoke_keeps_action_and_answer_before_and_after_semantic_frame(self) -> None:
        scenarios = [
            "Какой каталог мне нужен для Mercedes?",
            "Оплатил, но доступа нет",
            "Как купить подписку?",
        ]
        for user_query in scenarios:
            with self.subTest(user_query=user_query):
                legacy_app = build_app(self.project_root)
                explicit_app = build_app(self.project_root)

                legacy_app._intent_agent._context_understanding_agent = _MappedContextUnderstandingAgent(
                    explicit_semantic_frame=False
                )
                explicit_app._intent_agent._context_understanding_agent = _MappedContextUnderstandingAgent(
                    explicit_semantic_frame=True
                )
                legacy_app._intent_agent._topic_classifier = _QueryMappedTopicClassifier()
                explicit_app._intent_agent._topic_classifier = _QueryMappedTopicClassifier()
                legacy_app._response_agent._llm = _StaticAnswerLLM()
                explicit_app._response_agent._llm = _StaticAnswerLLM()

                legacy_response = legacy_app.respond(session_id=f"legacy-{self._testMethodName}-{user_query}", user_query=user_query)
                explicit_response = explicit_app.respond(session_id=f"explicit-{self._testMethodName}-{user_query}", user_query=user_query)

                self.assertEqual(legacy_response.action_name, explicit_response.action_name)
                self.assertEqual(legacy_response.topic_ids, explicit_response.topic_ids)
                self.assertEqual(legacy_response.answer_text, explicit_response.answer_text)


if __name__ == "__main__":
    unittest.main()
