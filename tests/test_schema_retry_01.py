from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.core.interfaces import LLMProvider
from src.core.models import SessionState
from src.prompting.prompt_manager import PromptManager
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.config.resource_paths import SEMANTIC_INTENTS_PATH, TOPIC_CLASSIFIER_PROMPT_PATH
from the_First_Agent.orchestrator.topic_classifier import TopicClassifier
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder


class _RetryFakeLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        return ""

    def generate_json(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return json.dumps({"label": "Тарифы TIS", "confidence": 0.95}, ensure_ascii=False)
        return json.dumps(
            {
                "intent_1": {
                    "intent_id": "tis_tariffs",
                    "score": 0.93,
                    "reason": "Клиент спрашивает стоимость TIS.",
                },
                "intent_2": None,
            },
            ensure_ascii=False,
        )


class _StaticJsonLLMProvider(LLMProvider):
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def generate_text(self, prompt: str) -> str:
        return ""

    def generate_json(self, prompt: str) -> str:
        return json.dumps(self.response, ensure_ascii=False)


class SchemaRetry01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            "prompts/answer_generator_prompt.txt",
        )
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.topic_prompt_sections_builder = TopicPromptSectionsBuilder()

    def _classify_direct(
        self,
        classifier: TopicClassifier,
        prompt: str,
        *,
        history_text: str,
        user_query: str,
        session_state: SessionState,
    ):
        return classifier.classify(
            prompt,
            dialog_text=self.prompt_manager.build_dialog_text(history_text=history_text, user_query=user_query),
            user_query=user_query,
            session_state=session_state,
        )

    def test_schema_mismatch_retries_and_recovers(self) -> None:
        fake_llm = _RetryFakeLLMProvider()
        classifier = TopicClassifier(
            llm_provider=fake_llm,
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="",
            user_query="сколько стоит TIS",
            session_state_json=json.dumps(SessionState().as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )

        result = self._classify_direct(
            classifier,
            prompt,
            history_text="",
            user_query="сколько стоит TIS",
            session_state=SessionState(),
        )

        self.assertEqual(result.topic_ids, ["tis_tariffs"])
        self.assertEqual(fake_llm.calls, 2)
        self.assertEqual(result.fallback_reason, "")
        self.assertEqual(result.diagnostics.get("validation_errors"), [])
        self.assertTrue(result.diagnostics.get("schema_retry_used"))
        attempts = result.diagnostics.get("llm_attempts", [])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].get("fallback_reason"), "schema_validation_failed")
        self.assertTrue(attempts[0].get("validation_errors"))
        self.assertEqual(attempts[1].get("fallback_reason"), "")
        self.assertEqual(len(fake_llm.prompts), 2)
        self.assertIn("Твоя предыдущая попытка нарушила формат JSON.", fake_llm.prompts[1])
        self.assertIn("intent_id", fake_llm.prompts[1])
        self.assertIn("не используй поля label, intent, confidence", fake_llm.prompts[1])

    def test_general_pricing_post_rule_adds_second_topic(self) -> None:
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "epc_tariffs",
                        "score": 0.93,
                        "reason": "Клиент спрашивает общую стоимость подписки.",
                    },
                    "intent_2": None,
                }
            ),
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="",
            user_query="сколько стоит подписка",
            session_state_json=json.dumps(SessionState().as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="",
            user_query="сколько стоит подписка",
            session_state=SessionState(),
        )
        self.assertEqual(result.topic_ids, ["epc_tariffs", "tis_tariffs"])
        self.assertTrue(result.diagnostics.get("pipeline_contract", {}).get("post_model_topic_adjustments_enabled"))
        self.assertEqual(
            result.diagnostics.get("post_rule_trace"),
            [
                {
                    "rule": "general_pricing_add_missing_tariff",
                    "before_topic_ids": ["epc_tariffs"],
                    "after_topic_ids": ["epc_tariffs", "tis_tariffs"],
                    "reason": "Общий вопрос о стоимости без уточнения EPC/TIS.",
                }
            ],
        )

    def test_general_pricing_ignores_stale_brand_without_anaphora(self) -> None:
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "epc_tariffs",
                        "score": 0.93,
                        "reason": "Клиент спрашивает общую стоимость подписки.",
                    },
                    "intent_2": None,
                }
            ),
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="user: есть Volvo?\nassistant: да, есть",
            user_query="сколько стоит подписка",
            session_state_json=json.dumps(SessionState(last_mentioned_brand="Volvo").as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="user: есть Volvo?\nassistant: да, есть",
            user_query="сколько стоит подписка",
            session_state=SessionState(last_mentioned_brand="Volvo"),
        )
        self.assertEqual(result.topic_ids, ["epc_tariffs", "tis_tariffs"])

    def test_general_pricing_post_rule_does_not_add_unknown_topic(self) -> None:
        available_topics = set(self.topic_catalog.topics.keys()) - {"tis_tariffs"}
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "epc_tariffs",
                        "score": 0.93,
                        "reason": "Клиент спрашивает общую стоимость подписки.",
                    },
                    "intent_2": None,
                }
            ),
            topic_ids=available_topics,
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="",
            user_query="сколько стоит подписка",
            session_state_json=json.dumps(SessionState().as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="",
            user_query="сколько стоит подписка",
            session_state=SessionState(),
        )
        self.assertEqual(result.topic_ids, ["epc_tariffs"])
        self.assertEqual(result.diagnostics.get("post_rule_trace"), [])

    def test_refund_post_rule_overrides_wrong_pricing_topics(self) -> None:
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "epc_tariffs",
                        "score": 0.85,
                        "reason": "Клиент спрашивает стоимость подписки.",
                    },
                    "intent_2": {
                        "intent_id": "tis_tariffs",
                        "score": 0.85,
                        "reason": "Клиент спрашивает стоимость TIS.",
                    },
                }
            ),
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="",
            user_query="можно вернуть деньги если не подошло",
            session_state_json=json.dumps(SessionState().as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="",
            user_query="можно вернуть деньги если не подошло",
            session_state=SessionState(),
        )
        self.assertEqual(result.topic_ids, ["post_payment_no_access"])
        self.assertEqual(
            result.diagnostics.get("post_rule_trace"),
            [
                {
                    "rule": "refund_override_pricing_topics",
                    "before_topic_ids": ["epc_tariffs", "tis_tariffs"],
                    "after_topic_ids": ["post_payment_no_access"],
                    "reason": "Запрос похож на возврат денег или post-payment проблему, поэтому pricing intent заменен.",
                }
            ],
        )

    def test_refund_post_rule_does_not_trigger_from_generic_access_wording(self) -> None:
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "epc_tariffs",
                        "score": 0.86,
                        "reason": "Клиент спрашивает стоимость подписки.",
                    },
                    "intent_2": None,
                }
            ),
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="",
            user_query="сколько стоит доступ",
            session_state_json=json.dumps(SessionState().as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="",
            user_query="сколько стоит доступ",
            session_state=SessionState(),
        )
        self.assertEqual(result.topic_ids, ["epc_tariffs", "tis_tariffs"])
        self.assertEqual(
            result.diagnostics.get("post_rule_trace"),
            [
                {
                    "rule": "general_pricing_add_missing_tariff",
                    "before_topic_ids": ["epc_tariffs"],
                    "after_topic_ids": ["epc_tariffs", "tis_tariffs"],
                    "reason": "Общий вопрос о стоимости без уточнения EPC/TIS.",
                }
            ],
        )

    def test_no_post_rule_yields_empty_trace(self) -> None:
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "demo_access",
                        "score": 0.91,
                        "reason": "Клиент спрашивает про демо-доступ.",
                    },
                    "intent_2": None,
                }
            ),
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="",
            user_query="а демо есть?",
            session_state_json=json.dumps(SessionState().as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="",
            user_query="а демо есть?",
            session_state=SessionState(),
        )
        self.assertEqual(result.topic_ids, ["demo_access"])
        self.assertEqual(result.diagnostics.get("post_rule_trace"), [])

    def test_brand_resolution_trace_uses_query_brand(self) -> None:
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "specific_brand_check",
                        "score": 0.92,
                        "reason": "Клиент спрашивает про конкретный бренд.",
                    },
                    "intent_2": None,
                }
            ),
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="",
            user_query="есть Volvo?",
            session_state_json=json.dumps(SessionState().as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="",
            user_query="есть Volvo?",
            session_state=SessionState(),
        )
        self.assertEqual(result.diagnostics.get("brand_resolution_trace"), {"source": "query", "brand": "Volvo"})
        self.assertEqual(result.state_snapshot.get("last_mentioned_brand"), "Volvo")

    def test_brand_resolution_trace_preserves_session_brand_without_using_query_brand(self) -> None:
        classifier = TopicClassifier(
            llm_provider=_StaticJsonLLMProvider(
                {
                    "intent_1": {
                        "intent_id": "epc_tariffs",
                        "score": 0.92,
                        "reason": "Клиент спрашивает общую стоимость подписки.",
                    },
                    "intent_2": None,
                }
            ),
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / "src/config/config_intents.yaml",
            brands_file_path=self.project_root / "src/config/brands.yaml",
        )
        session_state = SessionState(last_mentioned_brand="Volvo")
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(),
            topics_text=self.topic_catalog.as_prompt_text(),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(set(self.topic_catalog.topics.keys())),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(set(self.topic_catalog.topics.keys())),
            history_text="user: есть Volvo?\nassistant: да, есть",
            user_query="сколько стоит подписка?",
            session_state_json=json.dumps(session_state.as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(), ensure_ascii=False),
        )
        result = self._classify_direct(
            classifier,
            prompt,
            history_text="user: есть Volvo?\nassistant: да, есть",
            user_query="сколько стоит подписка?",
            session_state=session_state,
        )
        self.assertEqual(result.diagnostics.get("brand_resolution_trace"), {"source": "session", "brand": "Volvo"})
        self.assertEqual(result.state_snapshot.get("last_mentioned_brand"), "Volvo")


if __name__ == "__main__":
    unittest.main()
