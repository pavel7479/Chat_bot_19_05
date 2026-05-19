from __future__ import annotations

import json
import logging
import os
import sys
import unittest
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.loader import ConfigLoader
from src.core.models import ChatMessage, SessionState
from src.prompting.prompt_manager import PromptManager
from src.agents.intent_agent import IntentAgent
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.Agent_Zero.context_understanding_agent import ContextUnderstandingAgent
from the_First_Agent.config.resource_paths import CONTEXT_SIGNAL_RULES_PATH, SEMANTIC_INTENTS_PATH, TOPIC_CLASSIFIER_PROMPT_PATH
from the_First_Agent.context.context_signal_extractor import ContextSignalExtractor
from the_First_Agent.context.history_formatter import format_history
from the_First_Agent.orchestrator.topic_classifier import TopicClassifier
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder
from tests.support.reliable_ollama_provider import ReliableOllamaTestProvider, TestLLMRuntimeConfig


CASE_MATRIX: list[dict[str, object]] = [
    {
        "name": "01_help_capabilities",
        "history": [],
        "query": "Чем ты мне можешь помочь?",
        "expected_any": ["company_services_info"],
    },
    {
        "name": "02_generic_catalog_request",
        "history": [],
        "query": "мне нужен каталог, какие у вас есть?",
        "expected_any": ["brand_list_request", "company_services_info"],
    },
    {
        "name": "03_documents_request",
        "history": [],
        "query": "хочу ознакомиться с договором и получить карточку вашего предприятия",
        "expected_any": ["legal_entity_purchase_flow", "company_services_info"],
    },
    {
        "name": "04_subscription_price_dual",
        "history": [],
        "query": "сколько стоит подписка",
        "expected_all": ["epc_tariffs", "tis_tariffs"],
    },
    {
        "name": "05_catalog_list",
        "history": [],
        "query": "какие каталоги у вас есть",
        "expected_any": ["brand_list_request", "company_services_info"],
    },
    {
        "name": "06_specific_brand_volvo",
        "history": [],
        "query": "есть ли каталог вольво",
        "expected_any": ["specific_brand_check"],
    },
    {
        "name": "07_tis_followup_after_volvo",
        "history": [
            ("user", "есть ли каталог вольво"),
            ("assistant", "Да, каталог Volvo есть в наличии. Подскажите, пожалуйста, для каких моделей Volvo вас интересует каталог?"),
        ],
        "query": "а сколько TIS на него",
        "expected_any": ["tis_tariffs"],
    },
    {
        "name": "08_refund_after_purchase",
        "history": [],
        "query": "можно вернуть деньги если не подошло",
        "expected_any": ["post_payment_no_access", "company_services_info"],
    },
    {
        "name": "09_compare_catalogs",
        "history": [
            ("user", "какие каталоги у вас есть"),
            (
                "assistant",
                "У нас есть два направления: EPC Full — единый каталог запчастей по всем мировым брендам и видам техники, а TIS — отдельные технические базы и сервисные модули по автомобильным брендам. Если хотите, могу коротко объяснить, в чём разница между EPC и TIS.",
            ),
        ],
        "query": "в чём отличие",
        "expected_any": ["company_services_info"],
    },
    {
        "name": "10_subscription_advantages",
        "history": [],
        "query": "какие преимущества вашей подписки",
        "expected_any": ["company_services_info"],
    },
    {
        "name": "11_post_payment_access_timing",
        "history": [],
        "query": "после оплаты доступ сразу появиться",
        "expected_any": ["post_payment_access_timing"],
    },
    {
        "name": "12_shacman_alias_brand",
        "history": [],
        "query": "нужен каталог шахман",
        "expected_any": ["specific_brand_check"],
    },
    {
        "name": "13_multi_user",
        "history": [],
        "query": "можно ли пользоваться нескольким людям",
        "expected_any": ["company_services_info", "legal_entity_purchase_flow", "multi_device_access"],
    },
    {
        "name": "14_out_of_scope",
        "history": [],
        "query": "мы шьём одежду для редких пород рыб",
        "expected_any": ["out_of_scope_request"],
    },
    {
        "name": "15_ambiguous_price_short",
        "history": [],
        "query": "сколько стоит",
        "expected_all": ["epc_tariffs", "tis_tariffs"],
    },
    {
        "name": "16_tis_all_followup",
        "history": [
            ("user", "TIS"),
            ("assistant", "По TIS подскажите, пожалуйста, интересующие бренды. Я сразу посчитаю итоговую сумму."),
        ],
        "query": "Все",
        "expected_any": ["tis_tariffs", "brand_list_request"],
    },
    {
        "name": "17_short_catalog_followup",
        "history": [
            ("user", "какие каталоги у вас есть"),
            ("assistant", "Да, у нас есть каталоги по широкому списку брендов."),
        ],
        "query": "а какие есть",
        "expected_any": ["brand_list_request", "company_services_info"],
    },
    {
        "name": "18_checkout_pushback",
        "history": [],
        "query": "это куда ты меня оформлять собрался",
        "expected_any": ["company_services_info"],
    },
]


class TopicFirstAgent02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.tests_log_path = cls.project_root / "logs/tests_first_agent_02.log"
        cls.logger = logging.getLogger("topic_first_agent_02_tests")
        cls.logger.setLevel(logging.INFO)
        cls.logger.handlers.clear()
        cls.logger.propagate = False
        handler = logging.FileHandler(cls.tests_log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        cls.logger.addHandler(handler)
        cls.logger.info("===== TEST RUN START =====")
        config_path = Path(
            os.getenv(
                "CHATBOT_CONFIG_PATH",
                str(cls.project_root / "config.yaml"),
            )
        )
        config = ConfigLoader(config_path).load()
        runtime_cfg = yaml.safe_load((cls.project_root / "tests/config/test_llm_runtime.yaml").read_text(encoding="utf-8")) or {}
        test_llm = runtime_cfg.get("test_llm", {})
        runtime = TestLLMRuntimeConfig(
            per_request_timeout_s=int(test_llm.get("per_request_timeout_s", 35)),
            max_retries=int(test_llm.get("max_retries", 2)),
            retry_backoff_s=float(test_llm.get("retry_backoff_s", 1.0)),
        )
        llm_provider = ReliableOllamaTestProvider(config.llm, runtime)
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.topic_prompt_sections_builder = TopicPromptSectionsBuilder()
        cls.topic_shortlist_builder = TopicShortlistBuilder(cls.topic_catalog.topics, top_k=8)
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            "prompts/answer_generator_prompt.txt",
        )
        cls.classifier = TopicClassifier(
            llm_provider=llm_provider,
            topic_ids=set(cls.topic_catalog.topics.keys()),
            topic_titles_by_id=cls.topic_catalog.title_map(),
            intents_config_path=cls.project_root / "src/config/config_intents.yaml",
            brands_file_path=cls.project_root / "src/config/brands.yaml",
        )
        cls.context_understanding_agent = ContextUnderstandingAgent(
            llm=llm_provider,
            prompt_path=cls.project_root / config.paths.context_understanding_prompt,
            logger=cls.logger,
        )
        cls.context_signal_extractor = ContextSignalExtractor(CONTEXT_SIGNAL_RULES_PATH)
        cls.intent_agent = IntentAgent(
            topic_catalog=cls.topic_catalog,
            topic_shortlist_builder=cls.topic_shortlist_builder,
            topic_classifier=cls.classifier,
            prompt_manager=cls.prompt_manager,
            topic_prompt_sections_builder=cls.topic_prompt_sections_builder,
            context_understanding_agent=cls.context_understanding_agent,
            context_signal_extractor=cls.context_signal_extractor,
        )

    def _classify(self, history: list[tuple[str, str]], user_query: str):
        messages = [ChatMessage(role=role, text=text) for role, text in history]
        history_text = format_history(messages)
        session_state = SessionState()
        _prompt, result = self.intent_agent.classify(
            history_text=history_text,
            user_query=user_query,
            session_state=session_state,
        )
        return result

    def test_first_agent_case_matrix(self) -> None:
        for case in CASE_MATRIX:
            with self.subTest(case=case["name"]):
                result = self._classify(case.get("history", []), str(case["query"]))
                topics = list(result.topic_ids)
                expected_any = [str(item) for item in case.get("expected_any", [])]
                expected_all = [str(item) for item in case.get("expected_all", [])]
                status = "PASS"
                if expected_any and not any(topic in expected_any for topic in topics):
                    status = "FAIL"
                if expected_all and not all(topic in topics for topic in expected_all):
                    status = "FAIL"
                self.logger.info(
                    "case=%s | status=%s | expected_any=%s | expected_all=%s | query=%s | topics=%s | confidence=%.3f | reason=%s | shortlist_ids=%s | shortlist=%s | raw=%s | parsed=%s | validation_errors=%s",
                    case["name"],
                    status,
                    expected_any,
                    expected_all,
                    case["query"],
                    topics,
                    float(result.confidence),
                    str(result.reason).strip(),
                    json.dumps(result.diagnostics.get("shortlist_topic_ids", []), ensure_ascii=False),
                    json.dumps(result.diagnostics.get("shortlist", []), ensure_ascii=False),
                    json.dumps(result.diagnostics.get("raw_llm_response", ""), ensure_ascii=False),
                    json.dumps(result.diagnostics.get("parsed_json", {}), ensure_ascii=False),
                    json.dumps(result.diagnostics.get("validation_errors", []), ensure_ascii=False),
                )
                self.assertTrue(topics, f"{case['name']}: empty topic_ids")
                self.assertLessEqual(len(topics), 2, f"{case['name']}: too many topics {topics}")
                self.assertTrue(str(result.reason).strip(), f"{case['name']}: empty reason")
                self.assertIsInstance(result.diagnostics, dict, f"{case['name']}: diagnostics missing")
                self.assertTrue(
                    str(result.diagnostics.get("raw_llm_response", "")).strip(),
                    f"{case['name']}: raw_llm_response missing",
                )
                self.assertIsInstance(result.diagnostics.get("parsed_json", {}), dict, f"{case['name']}: parsed_json missing")

                if expected_any:
                    self.assertTrue(
                        any(topic in expected_any for topic in topics),
                        f"{case['name']}: expected any of {expected_any}, got {topics}",
                    )
                if expected_all:
                    for topic in expected_all:
                        self.assertIn(topic, topics, f"{case['name']}: expected topic {topic}, got {topics}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.info("===== TEST RUN END =====")
        for handler in cls.logger.handlers:
            handler.close()
        cls.logger.handlers.clear()


if __name__ == "__main__":
    unittest.main()

# cd /root/project/Chat_bot
# /root/project/.venv/bin/python -m unittest tests/test_topic_first_agent_02.py

# truncate -s 0 /root/project/Chat_bot/logs/tests_first_agent_02.log
