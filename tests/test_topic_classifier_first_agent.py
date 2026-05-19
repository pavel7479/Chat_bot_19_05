from __future__ import annotations

import logging
import os
import sys
import unittest
from pathlib import Path
import json
from collections import Counter

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
from src.testing.test_result_logger import TestResultLogger
from tests.support.reliable_ollama_provider import ReliableOllamaTestProvider, TestLLMRuntimeConfig


class TopicClassifierFirstAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.tests_log_path = cls.project_root / "logs/tests.log"
        cls.test_runtime_config_path = cls.project_root / "tests/config/test_llm_runtime.yaml"
        cls.logger = logging.getLogger("topic_classifier_tests")
        cls.logger.setLevel(logging.INFO)
        cls.logger.handlers.clear()
        cls.logger.propagate = False
        handler = logging.FileHandler(cls.tests_log_path, encoding="utf-8")
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
        runtime = cls._load_test_runtime_config(cls.test_runtime_config_path)

        # Real model, but with test-only transport settings:
        # non-stream, retry and strict timeout to avoid long hangs.
        llm_provider = ReliableOllamaTestProvider(config.llm, runtime)

        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.topic_prompt_sections_builder = TopicPromptSectionsBuilder()
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            "prompts/answer_generator_prompt.txt",
        )
        cls.topic_shortlist_builder = TopicShortlistBuilder(cls.topic_catalog.topics, top_k=8)
        cls.classifier = TopicClassifier(
            llm_provider,
            set(cls.topic_catalog.topics.keys()),
            topic_titles_by_id=cls.topic_catalog.title_map(),
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
        cls._fallback_reasons = Counter()
        cls._total_cases = 0
        cls._trace_completeness = Counter()
        cls._source_counts = Counter()
        cls._test_result_logger = TestResultLogger(cls.logger)
        cls.logger.info(
            "classifier_runtime | source_of_truth=%s | pipeline_version=%s",
            "model_json",
            "first_agent_direct_v1",
        )

    @staticmethod
    def _load_test_runtime_config(path: Path) -> TestLLMRuntimeConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        test_llm = raw.get("test_llm", {})
        return TestLLMRuntimeConfig(
            per_request_timeout_s=int(test_llm.get("per_request_timeout_s", 35)),
            max_retries=int(test_llm.get("max_retries", 2)),
            retry_backoff_s=float(test_llm.get("retry_backoff_s", 1.0)),
        )

    def _classify(self, history_lines: list[str], user_query: str):
        history_messages: list[ChatMessage] = []
        for i, line in enumerate(history_lines):
            text = str(line).strip()
            lowered = text.lower()
            if lowered.startswith("assistant:") or lowered.startswith("бот:"):
                role = "assistant"
                text = text.split(":", 1)[1].strip()
            elif lowered.startswith("user:") or lowered.startswith("клиент:"):
                role = "user"
                text = text.split(":", 1)[1].strip()
            else:
                role = "user" if i % 2 == 0 else "assistant"
            history_messages.append(ChatMessage(role=role, text=text))
        history_text = format_history(history_messages)
        session_state = SessionState()
        prompt, result = self.intent_agent.classify(
            history_text=history_text,
            user_query=user_query,
            session_state=session_state,
        )
        self.assertGreaterEqual(len(result.topic_ids), 1)
        self.assertLessEqual(len(result.topic_ids), 2)
        self.assertIn("context_understanding", result.diagnostics)
        self.assertIn("last_context_gist", result.state_snapshot)
        self.assertIn("last_context_meaning", result.state_snapshot)
        return result

    def test_tis_brand_names_with_typos_returns_tis_and_specific_brand(self) -> None:
        result = self._classify(
            history_lines=["Нужен TIS", "Назовите бренды для расчета"],
            user_query="тайота, лекссс и porsch",
        )
        self.assertEqual(result.topic_ids[0], "tis_tariffs")
        self.assertEqual(set(result.topic_ids), {"tis_tariffs", "specific_brand_check"})

        diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        final_prompt = str(diagnostics.get("final_prompt", ""))
        raw_response = str(diagnostics.get("raw_llm_response", ""))

        self.assertIn("тайота, лекссс и porsch", final_prompt)
        self.assertIn("Клиент перечислил бренды для расчёта TIS.", final_prompt)
        self.assertNotIn('"intent_id": "brand_list_request"', raw_response)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.info(
            "classifier_summary | total_cases=%s | classifier_sources=%s | fallback_reasons=%s | trace_completeness=%s",
            cls._total_cases,
            dict(cls._source_counts),
            dict(cls._fallback_reasons),
            dict(cls._trace_completeness),
        )
        cls._test_result_logger.emit_summary()
        cls.logger.info("===== TEST RUN END =====")
        for handler in cls.logger.handlers:
            handler.close()
        cls.logger.handlers.clear()


TEST_CASES: list[dict[str, object]] = [
    {"name": "01_simple_tis", "history": [], "query": "Сколько стоит TIS?", "expected": ["tis_tariffs"]},
    {"name": "02_simple_epc", "history": [], "query": "Какая цена EPC Full?", "expected": ["epc_tariffs"]},
    {"name": "03_simple_physical", "history": [], "query": "Хочу купить как физ лицо", "expected": ["physical_person_purchase"]},
    {"name": "04_simple_why_no_private", "history": [], "query": "Почему не продаете физлицам?", "expected": ["no_private_sales_reason", "physical_person_purchase"]},
    {"name": "05_simple_legal_flow", "history": [], "query": "Мы ИП, как оплатить счет?", "expected": ["legal_entity_purchase_flow"]},
    {"name": "06_simple_demo", "history": [], "query": "Можно демо-доступ?", "expected": ["demo_access"]},
    {"name": "07_simple_api", "history": [], "query": "Есть API интеграция с 1С?", "expected": ["api_integration"]},
    {"name": "08_simple_price_objection", "history": [], "query": "У вас дорого, у других дешевле", "expected": ["price_objection"]},
    {"name": "09_simple_competitor", "history": [], "query": "Тогда уйду к конкуренту", "expected": ["competitor_choice"]},
    {"name": "10_simple_partial_catalog", "history": [], "query": "Можно купить только один бренд?", "expected": ["partial_catalog_request"]},
    {"name": "11_simple_services", "history": [], "query": "Расскажите о компании и услугах", "expected": ["company_services_info"]},
    {"name": "12_simple_accuracy", "history": [], "query": "Данные точные? не устаревшие?", "expected": ["data_accuracy_doubt"]},
    {"name": "13_simple_free_compare", "history": [], "query": "Сравните с бесплатными каталогами поставщиков", "expected": ["free_catalog_comparison"]},
    {"name": "14_simple_no_details", "history": [], "query": "Оплачу, но ИНН не дам", "expected": ["payment_without_details", "legal_entity_purchase_flow"]},
    {"name": "15_simple_human", "history": [], "query": "Позовите человека, не хочу с роботом", "expected": ["human_operator_request"]},
    {"name": "16_simple_brand_list", "history": [], "query": "Какие бренды у вас есть?", "expected": ["brand_list_request"]},
    {"name": "17_simple_specific_brand", "history": [], "query": "Есть ли каталог BMW?", "expected": ["specific_brand_check"]},
    {"name": "18_simple_purchase_ready", "history": [], "query": "Все устраивает, как купить?", "expected": ["purchase_ready"]},
    {"name": "19_simple_macos", "history": [], "query": "На macOS есть софт?", "expected": ["macos_support"]},
    {"name": "20_simple_dual_tis_and_physical", "history": [], "query": "Сколько стоит TIS и работаете ли с физ. лицами?", "expected": ["tis_tariffs", "physical_person_purchase"]},
    {"name": "21_context_followup_physical", "history": ["Хочу купить доступ в каталог"], "query": "А если я физлицо?", "expected": ["physical_person_purchase"]},
    {"name": "22_context_followup_legal_buy", "history": ["Мы юрлицо, готовы оплатить"], "query": "Что нужно для счета?", "expected": ["legal_entity_purchase_flow"]},
    {"name": "23_context_followup_demo_physical", "history": ["Можно демо доступ?"], "query": "Я частное лицо", "expected_any": ["physical_person_purchase", "demo_access"]},
    {"name": "24_context_followup_tis_after_epc", "history": ["client: Сколько стоит EPC?", "assistant: По EPC Full могу подсказать тарифы на нужный период доступа."], "query": "А TIS отдельно сколько?", "expected": ["tis_tariffs"]},
    {"name": "25_context_multi_turn_competitor", "history": ["У вас дорого", "Понимаю", "Наверное уйду к конкуренту"], "query": "Есть финальная скидка?", "expected_any": ["price_objection", "competitor_choice"]},
    {"name": "26_context_multi_turn_api", "history": ["Нужна выгрузка на сайт"], "query": "И интеграция с 1С нужна", "expected": ["api_integration"]},
    {"name": "27_context_multi_turn_brands", "history": ["Какие бренды есть?"], "query": "А конкретно Audi есть?", "expected_any": ["specific_brand_check", "brand_list_request"]},
    {"name": "28_context_multi_turn_human", "history": ["Не понимаю цены"], "query": "Позовите менеджера", "expected": ["human_operator_request"]},
    {"name": "29_context_multi_turn_accuracy", "history": ["В интернете каталоги бесплатные"], "query": "У вас данные точно актуальные?", "expected_any": ["data_accuracy_doubt", "free_catalog_comparison"]},
    {"name": "30_context_multi_turn_ready_to_buy", "history": ["client: Мы ИП", "assistant: Работаем с ИП и юрлицами, могу помочь с оформлением доступа."], "query": "Ок, как оформить покупку", "expected": ["legal_entity_purchase_flow"]},
    {"name": "31_context_short_question", "history": ["Интересует TIS для BMW"], "query": "Цена?", "expected": ["tis_tariffs"]},
    {"name": "32_context_short_question_epc", "history": ["Нужен доступ к EPC"], "query": "Сколько в месяц", "expected": ["epc_tariffs"]},
    {"name": "33_context_short_question_demo", "history": ["Я представитель автосервиса"], "query": "Демо дадите?", "expected": ["demo_access"]},
    {"name": "34_context_short_question_private_reason", "history": ["client: Я физлицо", "assistant: Мы продаем доступ только юридическим лицам и ИП."], "query": "Почему нельзя?", "expected": ["no_private_sales_reason"]},
    {"name": "35_context_dual_price_and_private", "history": ["Интересует TIS"], "query": "И еще вы с частными лицами не работаете?", "expected_any": ["physical_person_purchase", "no_private_sales_reason"]},
    {"name": "36_context_dual_api_and_buy", "history": ["Готовы оплатить"], "query": "Но нужен API", "expected_any": ["api_integration", "purchase_ready"]},
    {"name": "37_context_dual_macos_and_buy", "history": ["Хочу купить доступ"], "query": "На маке софт есть?", "expected_any": ["macos_support", "purchase_ready"]},
    {"name": "38_context_dual_requisites", "history": ["Ок оплачиваем"], "query": "Без ИНН можно?", "expected": ["payment_without_details", "legal_entity_purchase_flow"]},
    {"name": "39_context_dual_partial_and_tis", "history": ["Можно только один бренд?"], "query": "Либо тогда только TIS", "expected": ["partial_catalog_request", "tis_tariffs"]},
    {"name": "40_context_unknown", "history": ["Добрый день"], "query": "Как погода в Москве?", "expected_any": ["out_of_scope_request", "nonsense_input"]},
    {"name": "41_ml_yes_legal_after_question", "history": ["Хочу купить доступ", "Мы работаем только с юридическими лицами (ИП относиться к юридическим лицам). Вы юр. лицо?"], "query": "Да, являюсь", "expected": ["legal_entity_purchase_flow"]},
    {"name": "42_ml_no_physical_after_question", "history": ["Хочу купить доступ", "Мы работаем только с юридическими лицами (ИП относиться к юридическим лицам). Вы юр. лицо?"], "query": "Неа, я физик", "expected": ["physical_person_purchase"]},
    {"name": "43_ml_yes_after_demo_legal", "history": ["Можно демо?", "Мы можем предоставить демо-доступ только юридическим лицам. Вы юридическим лицом или представителем автобизнеса?"], "query": "yes, legal", "expected": ["demo_access", "legal_entity_purchase_flow"]},
    {"name": "44_ml_no_after_demo", "history": ["Можно демо?", "Мы можем предоставить демо-доступ только юридическим лицам. Вы юридическим лицом или представителем автобизнеса?"], "query": "нет", "expected": ["demo_access", "physical_person_purchase"]},
    {"name": "45_ml_short_how_much_after_epc_answer", "history": ["Сколько стоит EPC?", "У нас единый тариф по каталогам запчастей — EPC Full. 1 месяц — 5500, 3 месяца — 15000, 6 месяцев — 28800, 12 месяцев — 54000."], "query": "а на год?", "expected": ["epc_tariffs"]},
    {"name": "46_ml_short_only_tis_after_epc", "history": ["Нужен доступ", "Мы продаем только EPC Full по месяцам."], "query": "ок, тогда только тис", "expected_any": ["tis_tariffs"]},
    {"name": "47_ml_requisites_followup", "history": ["Как купить?", "Пожалуйста, напишите ИНН вашей компании, телефон, имя, период и количество доступов."], "query": "ИНН позже дам, можно сейчас оплатить?", "expected": ["payment_without_details", "legal_entity_purchase_flow"]},
    {"name": "48_ml_card_or_invoice", "history": ["Хочу купить", "Пожалуйста, напишите ИНН вашей компании, телефон, имя, период и количество доступов. Также уточните удобный способ оплаты - выставить счёт на оплату или оплатить по карте (QR - код)."], "query": "давайте по qr", "expected": ["legal_entity_purchase_flow"]},
    {"name": "49_ml_price_objection_after_tariff", "history": ["Сколько стоит доступ?", "1 месяц — 5500 рублей"], "query": "дорого", "expected": ["price_objection"]},
    {"name": "50_ml_competitor_after_objection", "history": ["Это дорого", "Понимаю, цены могут отличаться, но мы отвечаем за качество и актуальность данных."], "query": "ладно, уйду к тем кто дешевле", "expected": ["competitor_choice"]},
    {"name": "51_ml_partial_then_yes_tis", "history": ["Можно только BMW каталог?", "Каталоги на отдельные бренды не продаются. Отдельно можно купить только TIS."], "query": "да, тогда tiss на bmw", "expected": ["tis_tariffs"]},
    {"name": "52_ml_brand_typo_english", "history": ["Какие бренды есть?", "Система охватывает все мировые бренды."], "query": "а Mercedec и Wolksvagen есть?", "expected": ["specific_brand_check"]},
    {"name": "53_ml_brand_typo_russian", "history": ["Какие бренды есть?", "Система охватывает все мировые бренды."], "query": "а тайота и лексус есть?", "expected": ["specific_brand_check"]},
    {"name": "54_ml_brand_typo_mix", "history": ["Какие бренды есть?", "Система охватывает все мировые бренды."], "query": "есть ли хундай/hyndai и киаа?", "expected": ["specific_brand_check"]},
    {"name": "55_ml_api_after_buying_intent", "history": ["Готовы купить доступ", "Отлично, напишите ИНН и реквизиты для счета."], "query": "а api в 1с можете подключить?", "expected_any": ["api_integration", "purchase_ready"]},
    {"name": "56_ml_no_api_acceptance", "history": ["Нужна интеграция API", "Извините, на данный момент у нас нет решения для интеграции по API."], "query": "ок, тогда без api, просто доступ", "expected_any": ["purchase_ready"]},
    {"name": "57_ml_human_request_after_long_chat", "history": ["Сколько стоит?", "У нас есть EPC и TIS тарифы", "Не уверен что подходит", "Я могу помочь быстрее и точнее, если нужно передам менеджеру."], "query": "позови человека", "expected": ["human_operator_request"]},
    {"name": "58_ml_services_after_prices", "history": ["Какая цена?", "EPC Full 5500 в месяц"], "query": "а что вообще умеет ваша система?", "expected": ["company_services_info"]},
    {"name": "59_ml_accuracy_after_vin_problem", "history": ["У конкурента vin не пробился", "У нас максимальное покрытие брендов по VIN-подбору."], "query": "а у вас точно не будет ошибок?", "expected": ["data_accuracy_doubt"]},
    {"name": "60_ml_free_catalog_compare_short", "history": ["Я пользуюсь бесплатными каталогами", "Бесплатные каталоги часто устаревшие и неполные."], "query": "чем вы лучше бесплатных?", "expected": ["free_catalog_comparison"]},
    {"name": "61_ml_demo_then_physical_reason", "history": ["Хочу демо", "Мы можем предоставить демо-доступ только юридическим лицам. Вы юридическим лицом или представителем автобизнеса?"], "query": "почему физикам нельзя?", "expected_any": ["no_private_sales_reason", "physical_person_purchase"]},
    {"name": "62_ml_short_yes_after_invoice", "history": ["client: Хочу купить доступ", "assistant: Для оформления счета нужен ИНН вашей компании. Вы ИП?"], "query": "да", "expected": ["legal_entity_purchase_flow"]},
    {"name": "63_ml_short_no_after_legal_question", "history": ["Хочу купить", "Вы юр. лицо?"], "query": "нет", "expected": ["physical_person_purchase"]},
    {"name": "64_ml_short_yep_after_legal_question", "history": ["Хочу купить", "Вы юр. лицо?"], "query": "ага", "expected": ["legal_entity_purchase_flow"]},
    {"name": "65_ml_macos_after_purchase_flow", "history": ["Готов купить доступ", "Напишите ИНН и телефон для выставления счета."], "query": "а на мак ос есть приложение?", "expected_any": ["macos_support", "purchase_ready"]},
    {"name": "66_ml_epc_vs_tis_clarification", "history": ["Нужен тис", "TIS — это отдельные технические базы, не входящие в EPC Full."], "query": "епс туда входит или отдельно?", "expected": ["product_relation_or_difference"]},
    {"name": "67_ml_brand_with_typos_cyrillic", "history": ["Какие каталоги есть?", "У нас есть каталоги по всем мировым брендам"], "query": "а бмв, мерседес, фольцваген, шкода есть?", "expected": ["specific_brand_check"]},
    {"name": "68_ml_brand_with_typos_latin", "history": ["Какие каталоги есть?", "У нас есть каталоги по всем мировым брендам"], "query": "do you have porshe, renauldt, pegeot?", "expected": ["specific_brand_check"]},
    {"name": "69_ml_partial_reject_then_tis_sum", "history": ["Можно только audi каталог?", "Отдельные бренды не продаются, отдельно можно купить только TIS."], "query": "ок, посчитай ауди тис и bmw тис", "expected_all": ["tis_tariffs"], "expected_any": ["partial_catalog_request", "specific_brand_check"]},
    {"name": "70_ml_requisites_missing_city", "history": ["Хочу оплатить", "Нужно название компании, ИНН и город."], "query": "город не скажу, оплату примите?", "expected_any": ["payment_without_details"]},
    {"name": "71_ml_competitor_then_return", "history": ["Уйду к конкуренту", "Часто экономия выходит дороже из-за неполных данных."], "query": "ладно, сколько у вас epc на 3 месяца?", "expected_any": ["epc_tariffs", "competitor_choice"]},
    {"name": "72_ml_api_then_operator", "history": ["Есть API?", "Сейчас API-интеграции нет, доступ только через наш сервер."], "query": "соедини с менеджером", "expected": ["human_operator_request", "api_integration"]},
    {"name": "73_ml_short_qualified_yes", "history": ["assistant: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?"], "query": "являюсь", "expected": ["demo_access", "legal_entity_purchase_flow"]},
    {"name": "74_ml_short_negative_nope", "history": ["assistant: Вы юридическое лицо?"], "query": "nope", "expected": ["physical_person_purchase"]},
    {"name": "75_ml_tis_brands_typo_combo", "history": ["Нужен TIS", "Назовите бренды для расчета"], "query": "тайота, лекссс и porsch", "expected": ["tis_tariffs", "specific_brand_check"]},
    {"name": "76_ml_objection_after_free_sources", "history": ["У поставщика бесплатно", "Бесплатные версии часто устаревшие и неполные."], "query": "все равно у вас дороже", "expected_any": ["price_objection", "free_catalog_comparison"]},
    {"name": "77_ml_buy_flow_after_private_reject", "history": ["Я физлицо", "Мы не продаем доступ частным лицам"], "query": "ок, оформим на ип, что нужно?", "expected_any": ["legal_entity_purchase_flow", "physical_person_purchase"]},
    {"name": "78_ml_tis_and_demo_mixed", "history": ["Интересует TIS", "Могу помочь подобрать бренды и стоимость"], "query": "сначала дайте демо", "expected_any": ["demo_access", "tis_tariffs"]},
    {"name": "79_ml_short_contextual_yes_to_invoice", "history": ["Подтвердите, пожалуйста: выставляем счет?", "Ответьте да или нет."], "query": "yes", "expected_any": ["legal_entity_purchase_flow"]},
    {"name": "80_ml_contextual_no_to_purchase", "history": ["Оформляем покупку сейчас?", "Напишите ИНН и телефон"], "query": "нет, пока изучаю", "expected": ["company_services_info"]},
    {"name": "81_mgr_price_subscription", "history": [], "query": "сколько стоит подписка", "expected_any": ["epc_tariffs", "tis_tariffs"]},
    {"name": "82_mgr_catalog_list", "history": [], "query": "какие каталоги у вас есть", "expected_any": ["brand_list_request", "company_services_info"]},
    {"name": "83_mgr_subscription_advantages", "history": [], "query": "какие преимущества вашей подписки", "expected_any": ["company_services_info", "epc_tariffs", "tis_tariffs"]},
    {"name": "84_mgr_compare_catalogs", "history": [], "query": "чем отличается один каталог от другого", "expected_any": ["company_services_info", "free_catalog_comparison"]},
    {"name": "85_mgr_which_catalog_mercedes", "history": [], "query": "какой каталог мне нужен для мерседес", "expected_any": ["specific_brand_check", "brand_list_request"]},
    {"name": "86_mgr_post_payment_access", "history": [], "query": "после оплаты доступ сразу появиться", "expected_any": ["post_payment_access_timing", "post_payment_no_access", "nonsense_input"]},
    {"name": "87_mgr_refund_policy", "history": [], "query": "можно вернуть деньги если не подошло", "expected_any": ["nonsense_input", "company_services_info", "post_payment_no_access"]},
    {"name": "88_mgr_followup_brand_after_generic", "history": ["какие каталоги у вас есть", "Да, у нас есть каталоги по широкому списку брендов. Уточните интересующую марку"], "query": "мерседес", "expected_any": ["specific_brand_check", "brand_list_request"]},
    {"name": "89_mgr_ambiguous_price", "history": [], "query": "сколько стоит", "expected_any": ["epc_tariffs", "tis_tariffs", "nonsense_input"]},
    {"name": "90_mgr_abuse_message", "history": [], "query": "ты тупишь", "expected_any": ["nonsense_input", "human_operator_request"]},
    {"name": "91_mgr_price_with_noise", "history": [], "query": "пришли информацию на год епс верхние лобки", "expected_any": ["epc_tariffs", "company_services_info"]},
    {"name": "92_mgr_unknown_brand_global_hawk", "history": [], "query": "каталог на global hawk есть? мой сбили хочу отремонтировать", "expected_any": ["out_of_scope_request", "specific_brand_check", "nonsense_input"]},
    {"name": "93_mgr_buy_and_price", "history": [], "query": "хочу доступ к каталогам что по деньгам", "expected_any": ["purchase_ready", "epc_tariffs", "tis_tariffs"]},
    {"name": "94_mgr_nonsense_dot", "history": [], "query": ".", "expected_any": ["nonsense_input"]},
    {"name": "95_mgr_nonsense_random", "history": [], "query": "мммммммаааааа 235", "expected_any": ["nonsense_input"]},
    {"name": "96_mgr_long_multi_intent", "history": [], "query": "здравствуйте я хочу разобраться какие каталоги у вас доступны сколько стоит подписка какие есть тарифы, можно ли оплатить по счёту или qr коду, можно ли подключить несколько сотрудников, есть ли тестовый период, чем отличаются епс от тис", "expected_any": ["brand_list_request", "epc_tariffs", "tis_tariffs", "free_catalog_comparison", "demo_access", "multi_device_access", "usage_limits"]},
    {"name": "97_mgr_abuse_moshenniki", "history": [], "query": "вы мошенники", "expected_any": ["nonsense_input", "human_operator_request"]},
    {"name": "98_mgr_free_access", "history": [], "query": "дай бесплатный доступ", "expected_any": ["demo_access", "nonsense_input"]},
    {"name": "99_mgr_free_forever", "history": [], "query": "мне обещали бесплатно навсегда", "expected_any": ["demo_access", "nonsense_input"]},
    {"name": "100_mgr_price_buy_login_combo", "history": [], "query": "сколько стоит мерседес, как купить и почему не работает вход", "expected_any": ["specific_brand_check", "epc_tariffs", "tis_tariffs", "purchase_ready", "post_payment_no_access", "nonsense_input"]},
    {"name": "101_mgr_subscription_price_catalog", "history": [], "query": "сколько стоит подписка на каталог", "expected_any": ["epc_tariffs", "tis_tariffs"]},
    {"name": "102_mgr_service_cons", "history": [], "query": "какие минусы у сервиса", "expected_any": ["company_services_info", "nonsense_input", "price_objection"]},
    {"name": "103_mgr_short_catalog_list", "history": [], "query": "ответь коротко какие каталоги есть", "expected_any": ["brand_list_request", "company_services_info"]},
    {"name": "104_mgr_detailed_subscription", "history": [], "query": "расскажи подробно про подписку", "expected_any": ["epc_tariffs", "tis_tariffs", "company_services_info"]},
    {"name": "105_mgr_cheaper_than_tis_context", "history": ["расскажи подробно про подписку", "Тариф EPC Full..."], "query": "а это дешевле чем тис", "expected_any": ["free_catalog_comparison", "epc_tariffs", "tis_tariffs"]},
    {"name": "106_mgr_multi_user_access", "history": [], "query": "можно ли пользоваться нескольким людям", "expected_any": ["usage_limits", "multi_device_access", "company_services_info", "legal_entity_purchase_flow"]},
    {"name": "107_mgr_catalog_loop_turn1", "history": [], "query": "какие есть каталоги", "expected_any": ["brand_list_request", "company_services_info"]},
    {"name": "108_mgr_catalog_loop_turn2", "history": ["какие есть каталоги", "Да, у нас есть каталоги по широкому списку брендов. Уточните интересующую марку"], "query": "какие марки есть каталоге", "expected_any": ["brand_list_request", "company_services_info"]},
    {"name": "109_mgr_catalog_loop_turn3", "history": ["какие есть каталоги", "Да, у нас есть каталоги по широкому списку брендов. Уточните интересующую марку", "какие марки есть каталоге", "Да, у нас есть каталоги по широкому списку брендов. Уточните интересующую марку"], "query": "каталоги каких марок есть", "expected_any": ["brand_list_request", "company_services_info"]},
    {"name": "110_mgr_brand_volvo", "history": [], "query": "есть ли каталог вольво", "expected_any": ["specific_brand_check", "brand_list_request"]},
    {"name": "111_mgr_brand_uaz", "history": [], "query": "есть ли каталог уаз", "expected_any": ["specific_brand_check", "brand_list_request"]},
]


def _build_test(case: dict[str, object]):
    def test_method(self: TopicClassifierFirstAgentTests) -> None:
        result = self._classify(
            history_lines=case["history"],
            user_query=case["query"],
        )
        diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        self.__class__.logger.info(
            "first_agent_case_log | case=%s | got=%s | raw=%s | parsed=%s | validation_errors=%s | fallback_reason=%s",
            case["name"],
            json.dumps(result.topic_ids, ensure_ascii=False),
            json.dumps(diagnostics.get("raw_llm_response", ""), ensure_ascii=False),
            json.dumps(diagnostics.get("parsed_json", {}), ensure_ascii=False),
            json.dumps(diagnostics.get("validation_errors", []), ensure_ascii=False),
            str(result.fallback_reason or ""),
        )
        topic_ids = result.topic_ids
        expected = case.get("expected", [])
        expected_any = case.get("expected_any", [])
        expected_all = case.get("expected_all", [])
        if expected:
            length_ok = len(topic_ids) == len(expected)
            topics_ok = sorted(topic_ids) == sorted(expected)
        else:
            length_ok = True
            topics_ok = True
            if expected_any:
                topics_ok = any(item in topic_ids for item in expected_any)
            if topics_ok and expected_all:
                topics_ok = all(item in topic_ids for item in expected_all)
        status = "PASS" if (length_ok and topics_ok) else "FAIL"
        self.logger.info(
            "case=%s | status=%s | expected=%s | expected_any=%s | expected_all=%s | got=%s | history=%s | query=%s | confidence=%.3f | reason=%s | diagnostics=%s | rule_trace=%s",
            case["name"],
            status,
            expected,
            expected_any,
            expected_all,
            topic_ids,
            case["history"],
            case["query"],
            result.confidence,
            result.reason,
            json.dumps(result.diagnostics, ensure_ascii=False),
            json.dumps(result.rule_trace, ensure_ascii=False),
        )
        state_trace = result.diagnostics.get("state_trace", [])
        trace_steps = {str(item.get("step", "")) for item in state_trace if isinstance(item, dict)}
        expected_trace_steps = {
            "receive_context",
            "normalize_input",
            "llm_generate_json",
            "parse_model_json",
            "build_state_snapshot",
        }
        trace_complete = expected_trace_steps.issubset(trace_steps)
        source = str(result.classifier_source or "unknown").strip() or "unknown"
        fallback_reason = str(result.fallback_reason or "").strip() or "none"
        final_topic = result.topic_ids[0] if result.topic_ids else "nonsense_input"

        self.__class__._total_cases += 1
        self.__class__._source_counts[source] += 1
        self.__class__._fallback_reasons[fallback_reason] += 1
        self.__class__._trace_completeness["complete" if trace_complete else "incomplete"] += 1

        self.logger.info(
            "routing_case_summary | case=%s | source=%s | fallback_reason=%s | final_topic=%s | trace_complete=%s",
            case["name"],
            source,
            fallback_reason,
            final_topic,
            trace_complete,
        )
        self.__class__._test_result_logger.record_case(
            case_name=str(case["name"]),
            status=status,
            fallback_reason=fallback_reason,
        )
        if expected:
            self.assertEqual(
                len(topic_ids),
                len(expected),
                msg=f"Unexpected topic count. Expected={expected}, got={topic_ids}",
            )
            self.assertCountEqual(
                topic_ids,
                expected,
                msg=f"Topic mismatch. Expected={expected}, got={topic_ids}",
            )
        else:
            if expected_any:
                self.assertTrue(
                    any(item in topic_ids for item in expected_any),
                    msg=f"Topic mismatch. Expected any of {expected_any}, got={topic_ids}",
                )
            if expected_all:
                self.assertTrue(
                    all(item in topic_ids for item in expected_all),
                    msg=f"Topic mismatch. Expected all of {expected_all}, got={topic_ids}",
                )
        self.assertTrue(
            isinstance(state_trace, list) and len(state_trace) > 0,
            msg=f"state_trace is empty for case={case['name']}",
        )
        self.assertTrue(
            any(str(item.get("step", "")) == "parse_model_json" for item in state_trace if isinstance(item, dict)),
            msg=f"state_trace has no parse_model_json step for case={case['name']}",
        )
        contract_info = result.diagnostics.get("pipeline_contract", {})
        self.assertTrue(
            isinstance(contract_info, dict) and bool(contract_info.get("source_of_truth", "")),
            msg=f"pipeline_contract.source_of_truth is missing for case={case['name']}",
        )
        self.assertTrue(
            isinstance(result.diagnostics.get("active_pipeline", []), list),
            msg=f"active_pipeline is missing for case={case['name']}",
        )
        self.assertTrue(
            result.classifier_source == "direct_json",
            msg=f"unexpected classifier_source for case={case['name']}: {result.classifier_source}",
        )
        self.assertTrue(
            trace_complete,
            msg=f"state_trace is incomplete for case={case['name']}: steps={sorted(trace_steps)}",
        )

        dropped_notes: list[str] = []
        for item in result.rule_trace:
            if not isinstance(item, dict):
                continue
            notes = item.get("notes", {})
            if not isinstance(notes, dict):
                continue
            for note in notes.values():
                text = str(note)
                if "dropped_by=" in text:
                    dropped_notes.append(text)
        if dropped_notes:
            self.assertTrue(
                all("dropped_by=" in note for note in dropped_notes),
                msg=f"dropped_by contract broken for case={case['name']}",
            )

    return test_method


for _case in TEST_CASES:
    setattr(
        TopicClassifierFirstAgentTests,
        f"test_first_agent_{_case['name']}",
        _build_test(_case),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

# cd /root/project/Chat_bot && CHATBOT_API_KEY="xK9mLpQ2vN7wR" /root/project/.venv/bin/python tests/test_topic_classifier_first_agent.py
# truncate -s 0 /root/project/Chat_bot/logs/tests.log
# cd /root/project/Chat_bot && /root/project/.venv/bin/python -m unittest tests/test_topic_classifier_first_agent.py
