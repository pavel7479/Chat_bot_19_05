from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from src.agents.intent_agent import IntentAgent
from src.config.loader import ConfigLoader
from src.core.models import SemanticFrame, SessionState, TopicClassificationResult
from src.intents.semantic_compatibility_validator import SemanticCompatibilityValidator
from src.prompting.prompt_manager import PromptManager
from the_First_Agent.Agent_Zero.models import ContextUnderstandingResult
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.config.resource_paths import (
    CONTEXT_SIGNAL_RULES_PATH,
    SEMANTIC_INTENTS_PATH,
    TOPIC_CLASSIFIER_PROMPT_PATH,
)
from the_First_Agent.context.context_signal_extractor import ContextSignalExtractor
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder


@dataclass(slots=True)
class _ShortlistCandidate:
    topic_id: str

    def as_dict(self) -> dict[str, object]:
        return {"topic_id": self.topic_id, "score": 1.0}


class _FakeShortlistBuilder:
    def __init__(self, shortlist_ids: list[str]) -> None:
        self._shortlist_ids = list(shortlist_ids)

    def build_shortlist(self, query: str, history_text: str = "", session_state=None, context_signals=None, top_k=None):
        del query, history_text, session_state, context_signals, top_k
        return [_ShortlistCandidate(topic_id=item) for item in self._shortlist_ids]

    def get_last_full_shortlist_scores(self):
        return [{"topic_id": item, "score": 1.0} for item in self._shortlist_ids]

    def get_last_selected_shortlist_scores(self):
        return [{"topic_id": item, "score": 1.0} for item in self._shortlist_ids]

    def get_last_semantic_routing_trace(self):
        return {"source": "fake_shortlist"}


class _FakeContextUnderstandingAgent:
    def __init__(self, result: ContextUnderstandingResult) -> None:
        self.result = result

    def understand(self, dialog_text: str, user_query: str) -> ContextUnderstandingResult:
        del dialog_text, user_query
        return self.result


class _FakeTopicClassifier:
    def __init__(self, topic_ids: list[str], reason: str = "stub") -> None:
        self._topic_ids = list(topic_ids)
        self._reason = reason

    def classify(
        self,
        prompt: str,
        *,
        dialog_text: str,
        user_query: str,
        context_signals=None,
        session_state: SessionState | None = None,
    ) -> TopicClassificationResult:
        del prompt, dialog_text, user_query, context_signals, session_state
        return TopicClassificationResult(
            topic_ids=list(self._topic_ids),
            confidence=0.83,
            reason=self._reason,
            diagnostics={
                "first_agent_output": {"state_snapshot": {}},
                "first_agent_trace": {"pipeline_steps": []},
                "state_trace": [],
            },
        )


class Stage02SemanticCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.validator = SemanticCompatibilityValidator(
            cls.project_root / "src/config/semantic_topic_map.yaml"
        )
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            cls.config.paths.answer_generator_prompt,
        )
        cls.sections_builder = TopicPromptSectionsBuilder()

    def _build_intent_agent(
        self,
        *,
        context_result: ContextUnderstandingResult,
        shortlist_ids: list[str],
        classifier_topic_ids: list[str],
    ) -> IntentAgent:
        return IntentAgent(
            topic_catalog=self.topic_catalog,
            topic_shortlist_builder=_FakeShortlistBuilder(shortlist_ids),
            topic_classifier=_FakeTopicClassifier(classifier_topic_ids),
            prompt_manager=self.prompt_manager,
            topic_prompt_sections_builder=self.sections_builder,
            context_understanding_agent=_FakeContextUnderstandingAgent(context_result),
            context_signal_extractor=ContextSignalExtractor(CONTEXT_SIGNAL_RULES_PATH),
        )

    def test_validator_marks_soft_conflict_for_product_recommendation_to_tis_tariffs(self) -> None:
        result = self.validator.validate(
            semantic_frame=SemanticFrame(
                conversation_mode="product_choice",
                user_goal="ask_product_recommendation",
                language="ru",
            ),
            topic_ids=["tis_tariffs"],
            shortlist_ids=["tis_tariffs", "specific_brand_check"],
            state=SessionState(),
        )

        self.assertFalse(result.is_compatible)
        self.assertEqual(result.severity, "SOFT_CONFLICT")
        self.assertTrue(result.override_applied)
        self.assertEqual(result.final_topic, "specific_brand_check")

    def test_validator_marks_hard_conflict_for_support_to_tis_tariffs(self) -> None:
        result = self.validator.validate(
            semantic_frame=SemanticFrame(
                conversation_mode="support",
                user_goal="ask_support",
                language="ru",
            ),
            topic_ids=["tis_tariffs"],
            shortlist_ids=["tis_tariffs", "post_payment_no_access"],
            state=SessionState(),
        )

        self.assertFalse(result.is_compatible)
        self.assertEqual(result.severity, "HARD_CONFLICT")
        self.assertTrue(result.override_applied)
        self.assertEqual(result.final_topic, "post_payment_no_access")

    def test_intent_agent_replaces_incompatible_topic_with_compatible_shortlist_topic(self) -> None:
        context_result = ContextUnderstandingResult(
            gist="Клиент хочет понять, какой каталог нужен для Mercedes.",
            meaning="Клиент спрашивает, какой продукт ему выбрать под задачу.",
            raw_response='{"gist":"x","meaning":"y"}',
            parsed_json={"gist": "x", "meaning": "y"},
            semantic_frame=SemanticFrame(
                conversation_mode="product_choice",
                user_goal="ask_product_recommendation",
                language="ru",
            ),
        )
        agent = self._build_intent_agent(
            context_result=context_result,
            shortlist_ids=["tis_tariffs", "specific_brand_check", "catalog_list_request"],
            classifier_topic_ids=["tis_tariffs"],
        )

        _, result = agent.classify(
            history_text="бот: Опишите задачу по каталогу.",
            user_query="Какой каталог мне нужен для Mercedes?",
            session_state=SessionState(),
        )

        self.assertEqual(result.topic_ids, ["specific_brand_check"])
        trace = result.diagnostics.get("semantic_compatibility_trace", {})
        self.assertEqual(trace.get("severity"), "SOFT_CONFLICT")
        self.assertTrue(trace.get("override_applied"))
        self.assertEqual(trace.get("final_topic"), "specific_brand_check")
        self.assertEqual(result.state_snapshot.get("last_primary_topic"), "specific_brand_check")

    def test_intent_agent_falls_back_to_nonsense_input_when_no_compatible_shortlist_topic(self) -> None:
        context_result = ContextUnderstandingResult(
            gist="Клиент сообщает о проблеме с доступом.",
            meaning="Клиент не может войти после оплаты.",
            raw_response='{"gist":"x","meaning":"y"}',
            parsed_json={"gist": "x", "meaning": "y"},
            semantic_frame=SemanticFrame(
                conversation_mode="support",
                user_goal="ask_support",
                language="ru",
            ),
        )
        agent = self._build_intent_agent(
            context_result=context_result,
            shortlist_ids=["tis_tariffs", "epc_tariffs"],
            classifier_topic_ids=["tis_tariffs"],
        )

        _, result = agent.classify(
            history_text="бот: Опишите проблему с доступом.",
            user_query="Оплатил, но доступа нет",
            session_state=SessionState(),
        )

        self.assertEqual(result.topic_ids, ["nonsense_input"])
        trace = result.diagnostics.get("semantic_compatibility_trace", {})
        self.assertEqual(trace.get("severity"), "HARD_CONFLICT")
        self.assertTrue(trace.get("fallback_topic_used"))
        self.assertEqual(trace.get("fallback_topic"), "nonsense_input")

    def test_dialog_act_path_marks_compatibility_as_skipped(self) -> None:
        context_result = ContextUnderstandingResult(
            gist="Клиент просит менеджера.",
            meaning="Клиент хочет связаться с менеджером.",
            raw_response='{"gist":"x","meaning":"y"}',
            parsed_json={"gist": "x", "meaning": "y"},
            semantic_frame=SemanticFrame(
                conversation_mode="manager",
                user_goal="ask_manager",
                language="ru",
            ),
        )
        agent = self._build_intent_agent(
            context_result=context_result,
            shortlist_ids=["human_operator_request"],
            classifier_topic_ids=["human_operator_request"],
        )

        class _Decision:
            applied = True
            topic_ids = ["human_operator_request"]
            action_name = "human_operator"
            reason = "dialog_act_router_applied"
            classifier_source = "dialog_act_router"
            trace = {"reason": "dialog_act_router_applied"}
            extra_state_patch = {}

        class _ProductContext:
            def as_dict(self):
                return {}

            mentioned_brands = []

        _, result = agent.build_from_dialog_act(
            history_text="бот: Подключить менеджера?",
            user_query="да, менеджера",
            decision=_Decision(),
            context_result=context_result,
            slot_trace={},
            product_context=_ProductContext(),
            session_state=SessionState(),
        )

        trace = result.diagnostics.get("semantic_compatibility_trace", {})
        self.assertTrue(trace.get("skipped"))
        self.assertEqual(trace.get("skip_reason"), "dialog_act_router_applied")


if __name__ == "__main__":
    unittest.main()
