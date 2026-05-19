from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from src.prompting.prompt_manager import PromptManager
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.config.resource_paths import SEMANTIC_INTENTS_PATH, TOPIC_CLASSIFIER_PROMPT_PATH
from the_First_Agent.prompting.topic_prompt_block_catalog import TopicPromptBlockCatalog
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder


class TopicPromptDynamic01Tests(unittest.TestCase):
    _BACKTICK_ID_RE = re.compile(r"`([a-z_]+)`")
    _JSON_INTENT_ID_RE = re.compile(r'"intent_id"\s*:\s*"([a-z_]+)"')

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            "prompts/answer_generator_prompt.txt",
        )
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.block_catalog = TopicPromptBlockCatalog()
        cls.topic_prompt_sections_builder = TopicPromptSectionsBuilder(cls.block_catalog)

    @classmethod
    def _extract_prompt_intent_ids(cls, text: str) -> set[str]:
        known = cls.block_catalog.known_intent_ids()
        found: set[str] = set()
        for pattern in (cls._BACKTICK_ID_RE, cls._JSON_INTENT_ID_RE):
            for match in pattern.findall(text):
                if match in known:
                    found.add(match)
        return found

    def test_prompt_block_catalog_has_no_undeclared_or_unknown_intents(self) -> None:
        self.assertEqual(self.block_catalog.validate_block_definitions(), [])

    def test_pricing_shortlist_prompt_excludes_unselected_intents(self) -> None:
        shortlist_ids = ["epc_tariffs", "tis_tariffs", "out_of_scope_request", "nonsense_input"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="сколько стоит подписка",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("epc_tariffs", prompt)
        self.assertIn("tis_tariffs", prompt)
        self.assertNotIn("legal_entity_purchase_flow", prompt)
        self.assertNotIn("human_operator_request", prompt)
        self.assertNotIn("demo_access", prompt)
        self.assertTrue(self._extract_prompt_intent_ids(prompt).issubset(set(shortlist_ids)))

    def test_legal_shortlist_prompt_excludes_pricing_examples(self) -> None:
        shortlist_ids = ["legal_entity_purchase_flow", "physical_person_purchase", "out_of_scope_request", "nonsense_input"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="user: Хочу купить доступ\nassistant: Вы юр. лицо?",
            user_query="Да, являюсь",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("legal_entity_purchase_flow", prompt)
        self.assertIn("Да, являюсь", prompt)
        self.assertNotIn("brand_list_request", prompt)
        self.assertNotIn("tis_tariffs", prompt)
        self.assertTrue(self._extract_prompt_intent_ids(prompt).issubset(set(shortlist_ids)))

    def test_topic_descriptions_do_not_reference_foreign_intents(self) -> None:
        shortlist_ids = ["epc_tariffs", "out_of_scope_request", "nonsense_input"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="сколько стоит epc",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("epc_tariffs", prompt)
        self.assertNotIn("brand_list_request", prompt)
        self.assertNotIn("legal_entity_purchase_flow", prompt)
        self.assertTrue(self._extract_prompt_intent_ids(prompt).issubset(set(shortlist_ids)))
        self.assertIn('- "сколько стоит EPC" → pricing', prompt)
        self.assertIn('- "цена EPC" → pricing', prompt)
        self.assertIn('- "стоимость подписки" без бренда → pricing', prompt)
        self.assertNotIn("choose_when:\n- -", prompt)

    def test_pricing_prompt_keeps_selected_cross_reference(self) -> None:
        shortlist_ids = ["epc_tariffs", "tis_tariffs", "out_of_scope_request", "nonsense_input"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="сколько стоит подписка",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("`tis_tariffs`", prompt)
        self.assertTrue(self._extract_prompt_intent_ids(prompt).issubset(set(shortlist_ids)))

    def test_refund_shortlist_prompt_has_no_foreign_intent_ids(self) -> None:
        shortlist_ids = [
            "payment_without_details",
            "post_payment_no_access",
            "epc_tariffs",
            "physical_person_purchase",
            "out_of_scope_request",
            "nonsense_input",
            "no_private_sales_reason",
            "purchase_ready",
        ]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="можно вернуть деньги если не подошло",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        extracted_ids = self._extract_prompt_intent_ids(prompt)
        self.assertTrue(extracted_ids.issubset(set(shortlist_ids)), msg=f"foreign ids in prompt: {sorted(extracted_ids - set(shortlist_ids))}")
        self.assertNotIn("tis_tariffs", prompt)
        self.assertNotIn("brand_list_request", prompt)
        self.assertNotIn("demo_access", prompt)
        self.assertNotIn("specific_brand_check", prompt)
        self.assertNotIn("legal_entity_purchase_flow", prompt)

    def test_prompt_excludes_unselected_fallback_intents_from_static_template(self) -> None:
        shortlist_ids = [
            "tis_tariffs",
            "partial_catalog_request",
            "no_private_sales_reason",
            "specific_brand_check",
            "epc_tariffs",
            "purchase_ready",
            "api_integration",
            "company_services_info",
        ]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="user: Можно только один бренд?",
            user_query="Либо тогда только TIS",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        extracted_ids = self._extract_prompt_intent_ids(prompt)
        self.assertTrue(
            extracted_ids.issubset(set(shortlist_ids)),
            msg=f"foreign ids in prompt: {sorted(extracted_ids - set(shortlist_ids))}",
        )
        self.assertNotIn("nonsense_input", prompt)
        self.assertNotIn("out_of_scope_request", prompt)

    def test_prompt_manager_filters_foreign_dynamic_rules_and_examples(self) -> None:
        shortlist_ids = ["post_payment_no_access", "epc_tariffs", "out_of_scope_request", "nonsense_input"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=(
                "Правила для текущего shortlist:\n"
                "- Возвращай `post_payment_no_access`.\n"
                "- Возвращай `brand_list_request`.\n\n"
                "Приоритеты для текущего shortlist:\n"
                "1. `epc_tariffs` важнее.\n"
                "2. `demo_access` важнее."
            ),
            dynamic_examples_text=(
                "Пример:\n"
                "{\n"
                "  \"intent_1\": {\n"
                "    \"intent_id\": \"post_payment_no_access\",\n"
                "    \"score\": 0.91,\n"
                "    \"reason\": \"ok\"\n"
                "  },\n"
                "  \"intent_2\": null\n"
                "}\n\n"
                "Пример:\n"
                "{\n"
                "  \"intent_1\": {\n"
                "    \"intent_id\": \"brand_list_request\",\n"
                "    \"score\": 0.91,\n"
                "    \"reason\": \"bad\"\n"
                "  },\n"
                "  \"intent_2\": null\n"
                "}"
            ),
            history_text="",
            user_query="можно вернуть деньги",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        extracted_ids = self._extract_prompt_intent_ids(prompt)
        self.assertTrue(extracted_ids.issubset(set(shortlist_ids)), msg=f"foreign ids in prompt: {sorted(extracted_ids - set(shortlist_ids))}")
        self.assertNotIn("brand_list_request", prompt)
        self.assertNotIn("demo_access", prompt)

    def test_switch_rule_and_example_appear_only_when_both_intents_are_allowed(self) -> None:
        shortlist_ids = ["tis_tariffs", "partial_catalog_request", "epc_tariffs", "specific_brand_check"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="user: Можно только один бренд?",
            user_query="Тогда сколько TIS?",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("переключается на TIS", prompt)
        self.assertIn("Клиент переключился на обсуждение TIS.", prompt)

        shortlist_ids_without_partial = ["tis_tariffs", "epc_tariffs", "specific_brand_check"]
        prompt_without_partial = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids_without_partial),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids_without_partial),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids_without_partial),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids_without_partial),
            history_text="user: Можно только один бренд?",
            user_query="Тогда сколько TIS?",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids_without_partial), ensure_ascii=False),
        )
        self.assertNotIn("переключается на TIS", prompt_without_partial)
        self.assertNotIn("Клиент переключился на обсуждение TIS.", prompt_without_partial)

    def test_prompt_contains_json_hardening(self) -> None:
        shortlist_ids = ["epc_tariffs", "tis_tariffs", "out_of_scope_request", "nonsense_input"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="сколько стоит",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("Формат ответа:", prompt)
        self.assertIn("\"intent_1\"", prompt)
        self.assertIn("\"intent_2\"", prompt)

    def test_dynamic_examples_keep_multiline_structure(self) -> None:
        shortlist_ids = ["company_services_info", "epc_tariffs"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="что умеет сервис",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("Последняя реплика клиента:\nчто умеет сервис\n\nОтвет:\n{", prompt)
        self.assertNotIn("что умеет сервис Ответ:", prompt)

    def test_prompt_rules_are_not_needlessly_duplicated(self) -> None:
        shortlist_ids = ["epc_tariffs", "tis_tariffs"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="сколько стоит",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn("Правила:", prompt)
        self.assertEqual(prompt.count("Правила:"), 1)
        self.assertNotIn("Обязательные правила:", prompt)
        self.assertEqual(prompt.count("Не придумывай intent_id вне разрешённого списка."), 1)

    def test_context_understanding_block_is_textual_and_positioned_near_user_query(self) -> None:
        shortlist_ids = ["legal_entity_purchase_flow", "physical_person_purchase", "out_of_scope_request", "nonsense_input"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="user: Можно демо?\nassistant: Вы юрлицо?",
            user_query="Да, являюсь",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
            context_understanding_text=PromptManager.build_context_understanding_text(
                gist="Клиент интересуется демо-доступом.",
                meaning="Клиент подтвердил юридический статус.",
            ),
        )
        self.assertIn("Последняя реплика клиента:", prompt)
        self.assertNotIn("Перефразированная последняя фраза клиента:", prompt)
        self.assertIn("- Суть диалога: Клиент интересуется демо-доступом.", prompt)
        self.assertIn("- Смысл последней реплики: Клиент подтвердил юридический статус.", prompt)
        self.assertNotIn('{"gist"', prompt)
        self.assertIn("`reason` должен ссылаться на последнюю реплику клиента или на блок промежуточного понимания.", prompt)
        self.assertIn("последнюю реплику клиента", prompt)
        self.assertIn("промежуточное понимание диалога", prompt)
        last_phrase_pos = prompt.index("Последняя реплика клиента:")
        context_pos = prompt.index("Промежуточное понимание диалога:")
        allowed_pos = prompt.index("Разрешённые intent_id:")
        self.assertLess(last_phrase_pos, context_pos)
        self.assertLess(context_pos, allowed_pos)
        self.assertIn("Не используй markdown.", prompt)
        self.assertIn("Не добавляй текст вне JSON.", prompt)

    def test_topic_rules_are_rendered_as_separate_points_without_collapsing(self) -> None:
        shortlist_ids = ["brand_list_request", "company_services_info", "demo_access"]
        prompt = self.prompt_manager.build_topic_prompt(
            allowed_intents_text=self.topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self.topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self.topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self.topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text="",
            user_query="какие бренды есть",
            session_state_json="{}",
            topic_title_map_json=json.dumps(self.topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
        )
        self.assertIn('intent_id: brand_list_request', prompt)
        self.assertIn('- "какие бренды есть" → `brand_list_request`', prompt)
        self.assertIn('- "а какие есть", "все", "еще какие" после вопроса про бренды → `brand_list_request`', prompt)
        self.assertIn('- "по грузовым" после вопроса про бренды → `brand_list_request`', prompt)
        self.assertNotIn('`brand_list_request` "а какие есть"', prompt)

    def test_topic_catalog_preserves_aliases_from_yaml(self) -> None:
        self.assertIn("епс", self.topic_catalog.topics["epc_tariffs"].aliases)
        self.assertIn("epc full", self.topic_catalog.topics["epc_tariffs"].aliases)
        self.assertIn("тис", self.topic_catalog.topics["tis_tariffs"].aliases)

    def test_topic_catalog_max_examples_limit_is_configurable(self) -> None:
        limited_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH, max_examples_per_topic=2)
        prompt_text = limited_catalog.as_prompt_text(["epc_tariffs"])
        examples_block = prompt_text.split("examples:\n", 1)[1]
        self.assertEqual(examples_block.count('- "'), 2)
        self.assertIn('- "сколько стоит EPC"', examples_block)
        self.assertIn('- "цена EPC"', examples_block)
        self.assertNotIn('- "стоимость подписки"', examples_block)


if __name__ == "__main__":
    unittest.main()
