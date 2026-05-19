from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.models import ChatMessage, SessionState
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.config.resource_paths import SEMANTIC_INTENTS_PATH
from the_First_Agent.context.history_formatter import format_history


@dataclass(slots=True)
class ShortlistCase:
    name: str
    history: list[str]
    query: str
    must_include: list[str] = field(default_factory=list)
    any_of: list[str] = field(default_factory=list)
    max_ranks: dict[str, int] = field(default_factory=dict)


SHORTLIST_CASES: list[ShortlistCase] = [
    ShortlistCase("01_simple_tis", [], "Сколько стоит TIS?", must_include=["tis_tariffs"], max_ranks={"tis_tariffs": 2}),
    ShortlistCase("02_simple_epc", [], "Какая цена EPC Full?", must_include=["epc_tariffs"], max_ranks={"epc_tariffs": 2}),
    ShortlistCase("03_simple_physical", [], "Хочу купить как физ лицо", must_include=["physical_person_purchase"], max_ranks={"physical_person_purchase": 3}),
    ShortlistCase("04_simple_why_no_private", [], "Почему не продаете физлицам?", must_include=["no_private_sales_reason", "physical_person_purchase"], max_ranks={"no_private_sales_reason": 3}),
    ShortlistCase("05_simple_legal_flow", [], "Мы ИП, как оплатить счет?", must_include=["legal_entity_purchase_flow"], max_ranks={"legal_entity_purchase_flow": 3}),
    ShortlistCase("06_simple_demo", [], "Можно демо-доступ?", must_include=["demo_access"], max_ranks={"demo_access": 3}),
    ShortlistCase("07_simple_api", [], "Есть API интеграция с 1С?", must_include=["api_integration"], max_ranks={"api_integration": 3}),
    ShortlistCase("08_simple_price_objection", [], "У вас дорого, у других дешевле", must_include=["price_objection"], max_ranks={"price_objection": 3}),
    ShortlistCase("09_simple_competitor", [], "Тогда уйду к конкуренту", must_include=["competitor_choice"], max_ranks={"competitor_choice": 3}),
    ShortlistCase("10_simple_partial_catalog", [], "Можно купить только один бренд?", must_include=["partial_catalog_request"], max_ranks={"partial_catalog_request": 3}),
    ShortlistCase("11_simple_services", [], "Расскажите о компании и услугах", must_include=["company_services_info"], max_ranks={"company_services_info": 3}),
    ShortlistCase("12_simple_accuracy", [], "Данные точные? не устаревшие?", must_include=["data_accuracy_doubt"], max_ranks={"data_accuracy_doubt": 3}),
    ShortlistCase("13_simple_free_compare", [], "Сравните с бесплатными каталогами поставщиков", must_include=["free_catalog_comparison"], max_ranks={"free_catalog_comparison": 3}),
    ShortlistCase("14_simple_no_details", [], "Оплачу, но ИНН не дам", must_include=["payment_without_details", "legal_entity_purchase_flow"], max_ranks={"payment_without_details": 3, "legal_entity_purchase_flow": 5}),
    ShortlistCase("15_simple_human", [], "Позовите человека, не хочу с роботом", must_include=["human_operator_request"], max_ranks={"human_operator_request": 3}),
    ShortlistCase("16_simple_brand_list", [], "Какие бренды у вас есть?", must_include=["brand_list_request"], max_ranks={"brand_list_request": 3}),
    ShortlistCase("17_simple_specific_brand", [], "Есть ли каталог BMW?", must_include=["specific_brand_check"], max_ranks={"specific_brand_check": 3}),
    ShortlistCase("18_simple_purchase_ready", [], "Все устраивает, как купить?", must_include=["purchase_ready"], max_ranks={"purchase_ready": 3}),
    ShortlistCase("19_simple_macos", [], "На macOS есть софт?", must_include=["macos_support"], max_ranks={"macos_support": 3}),
    ShortlistCase("20_simple_dual_tis_and_physical", [], "Сколько стоит TIS и работаете ли с физ. лицами?", must_include=["tis_tariffs", "physical_person_purchase"], max_ranks={"tis_tariffs": 3, "physical_person_purchase": 5}),
    ShortlistCase("21_context_followup_physical", ["Хочу купить доступ в каталог"], "А если я физлицо?", must_include=["physical_person_purchase"], max_ranks={"physical_person_purchase": 3}),
    ShortlistCase("22_context_followup_legal_buy", ["Мы юрлицо, готовы оплатить"], "Что нужно для счета?", must_include=["legal_entity_purchase_flow"], max_ranks={"legal_entity_purchase_flow": 3}),
    ShortlistCase("23_context_followup_demo_physical", ["Можно демо доступ?"], "Я частное лицо", must_include=["demo_access", "physical_person_purchase"], max_ranks={"physical_person_purchase": 4}),
    ShortlistCase("24_context_followup_tis_after_epc", ["Сколько стоит EPC?"], "А TIS отдельно сколько?", must_include=["tis_tariffs", "epc_tariffs"], max_ranks={"tis_tariffs": 3, "epc_tariffs": 5}),
    ShortlistCase("25_context_multi_turn_competitor", ["У вас дорого", "Понимаю", "Наверное уйду к конкуренту"], "Есть финальная скидка?", must_include=["price_objection"], max_ranks={"price_objection": 4}),
    ShortlistCase("26_context_multi_turn_api", ["Нужна выгрузка на сайт"], "И интеграция с 1С нужна", must_include=["api_integration"], max_ranks={"api_integration": 3}),
    ShortlistCase("27_context_multi_turn_brands", ["Какие бренды есть?"], "А конкретно Audi есть?", must_include=["specific_brand_check", "brand_list_request"], max_ranks={"specific_brand_check": 4}),
    ShortlistCase("28_context_multi_turn_human", ["Не понимаю цены"], "Позовите менеджера", must_include=["human_operator_request"], max_ranks={"human_operator_request": 3}),
    ShortlistCase("29_context_multi_turn_accuracy", ["В интернете каталоги бесплатные"], "У вас данные точно актуальные?", must_include=["data_accuracy_doubt"], max_ranks={"data_accuracy_doubt": 4}),
    ShortlistCase("30_context_multi_turn_ready_to_buy", ["Мы ИП", "Сколько стоит EPC?"], "Ок, как оформить покупку", must_include=["purchase_ready", "legal_entity_purchase_flow"], max_ranks={"purchase_ready": 4, "legal_entity_purchase_flow": 5}),
    ShortlistCase("31_context_short_question", ["Интересует TIS для BMW"], "Цена?", must_include=["tis_tariffs"], max_ranks={"tis_tariffs": 3}),
    ShortlistCase("32_context_short_question_epc", ["Нужен доступ к EPC"], "Сколько в месяц", must_include=["epc_tariffs"], max_ranks={"epc_tariffs": 3}),
    ShortlistCase("33_context_short_question_demo", ["Я представитель автосервиса"], "Демо дадите?", must_include=["demo_access"], max_ranks={"demo_access": 3}),
    ShortlistCase("34_context_short_question_private_reason", ["Я физлицо"], "Почему нельзя?", must_include=["no_private_sales_reason", "physical_person_purchase"], max_ranks={"no_private_sales_reason": 4}),
    ShortlistCase("35_context_dual_price_and_private", ["Интересует TIS"], "И еще вы с частными лицами не работаете?", must_include=["physical_person_purchase", "no_private_sales_reason"], max_ranks={"physical_person_purchase": 5, "no_private_sales_reason": 5}),
    ShortlistCase("36_context_dual_api_and_buy", ["Готовы оплатить"], "Но нужен API", must_include=["api_integration"], max_ranks={"api_integration": 4}),
    ShortlistCase("37_context_dual_macos_and_buy", ["Хочу купить доступ"], "На маке софт есть?", must_include=["macos_support"], max_ranks={"macos_support": 4}),
    ShortlistCase("38_context_dual_requisites", ["Ок оплачиваем"], "Без ИНН можно?", must_include=["payment_without_details", "legal_entity_purchase_flow"], max_ranks={"payment_without_details": 3, "legal_entity_purchase_flow": 5}),
    ShortlistCase("39_context_dual_partial_and_tis", ["Можно только один бренд?"], "Либо тогда только TIS", must_include=["partial_catalog_request", "tis_tariffs"], max_ranks={"partial_catalog_request": 5, "tis_tariffs": 4}),
    ShortlistCase("40_context_unknown", ["Добрый день"], "Как погода в Москве?", any_of=["out_of_scope_request", "nonsense_input"]),
    ShortlistCase("41_ml_yes_legal_after_question", ["Хочу купить доступ", "Мы работаем только с юридическими лицами (ИП относиться к юридическим лицам). Вы юр. лицо?"], "Да, являюсь", must_include=["legal_entity_purchase_flow"], max_ranks={"legal_entity_purchase_flow": 4}),
    ShortlistCase("42_ml_no_physical_after_question", ["Хочу купить доступ", "Мы работаем только с юридическими лицами (ИП относиться к юридическим лицам). Вы юр. лицо?"], "Неа, я физик", must_include=["physical_person_purchase"], max_ranks={"physical_person_purchase": 4}),
    ShortlistCase("43_ml_yes_after_demo_legal", ["Можно демо?", "Мы можем предоставить демо-доступ только юридическим лицам. Вы юридическим лицом или представителем автобизнеса?"], "yes, legal", must_include=["demo_access", "legal_entity_purchase_flow"], max_ranks={"demo_access": 5, "legal_entity_purchase_flow": 5}),
    ShortlistCase("44_ml_no_after_demo", ["Можно демо?", "Мы можем предоставить демо-доступ только юридическим лицам. Вы юридическим лицом или представителем автобизнеса?"], "нет", must_include=["demo_access", "physical_person_purchase"], max_ranks={"demo_access": 5, "physical_person_purchase": 5}),
    ShortlistCase("45_ml_short_how_much_after_epc_answer", ["Сколько стоит EPC?", "У нас единый тариф по каталогам запчастей — EPC Full. 1 месяц — 5500, 3 месяца — 15000, 6 месяцев — 28800, 12 месяцев — 54000."], "а на год?", must_include=["epc_tariffs"], max_ranks={"epc_tariffs": 3}),
    ShortlistCase("46_ml_short_only_tis_after_epc", ["Нужен доступ", "Мы продаем только EPC Full по месяцам."], "ок, тогда только тис", must_include=["tis_tariffs"], max_ranks={"tis_tariffs": 4}),
    ShortlistCase("47_ml_requisites_followup", ["Как купить?", "Пожалуйста, напишите ИНН вашей компании, телефон, имя, период и количество доступов."], "ИНН позже дам, можно сейчас оплатить?", must_include=["payment_without_details", "legal_entity_purchase_flow"], max_ranks={"payment_without_details": 4, "legal_entity_purchase_flow": 5}),
    ShortlistCase("48_ml_card_or_invoice", ["Хочу купить", "Пожалуйста, напишите ИНН вашей компании, телефон, имя, период и количество доступов. Также уточните удобный способ оплаты - выставить счёт на оплату или оплатить по карте (QR - код)."], "давайте по qr", must_include=["legal_entity_purchase_flow"], max_ranks={"legal_entity_purchase_flow": 4}),
    ShortlistCase("49_ml_price_objection_after_tariff", ["Сколько стоит доступ?", "1 месяц — 5500 рублей"], "дорого", must_include=["price_objection"], max_ranks={"price_objection": 3}),
    ShortlistCase("50_ml_competitor_after_objection", ["Это дорого", "Понимаю, цены могут отличаться, но мы отвечаем за качество и актуальность данных."], "ладно, уйду к тем кто дешевле", must_include=["competitor_choice"], max_ranks={"competitor_choice": 3}),
]

assert len(SHORTLIST_CASES) == 50


class TopicShortlistBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.shortlist_builder = TopicShortlistBuilder(cls.topic_catalog.topics, top_k=8)

    @staticmethod
    def _build_history_text(history_lines: list[str]) -> str:
        messages = [
            ChatMessage(role=("user" if index % 2 == 0 else "assistant"), text=line)
            for index, line in enumerate(history_lines)
        ]
        return format_history(messages)

    def _build_shortlist(self, case: ShortlistCase):
        history_text = self._build_history_text(case.history)
        return self.shortlist_builder.build_shortlist(
            case.query,
            history_text=history_text,
            session_state=SessionState(),
        )

    def _assert_case(self, case: ShortlistCase) -> None:
        shortlist = self._build_shortlist(case)
        shortlist_ids = [item.topic_id for item in shortlist]
        rank_by_topic = {topic_id: index + 1 for index, topic_id in enumerate(shortlist_ids)}

        self.assertLessEqual(len(shortlist_ids), 8, msg=f"{case.name}: shortlist longer than top_k=8 -> {shortlist_ids}")
        self.assertEqual(len(shortlist_ids), len(set(shortlist_ids)), msg=f"{case.name}: shortlist contains duplicates -> {shortlist_ids}")

        for topic_id in case.must_include:
            self.assertIn(
                topic_id,
                shortlist_ids,
                msg=(
                    f"{case.name}: expected shortlist to include '{topic_id}' for query {case.query!r}. "
                    f"Got {shortlist_ids}"
                ),
            )

        if case.any_of:
            self.assertTrue(
                any(topic_id in shortlist_ids for topic_id in case.any_of),
                msg=(
                    f"{case.name}: expected shortlist to include at least one of {case.any_of} "
                    f"for query {case.query!r}. Got {shortlist_ids}"
                ),
            )

        for topic_id, max_rank in case.max_ranks.items():
            self.assertIn(
                topic_id,
                rank_by_topic,
                msg=f"{case.name}: cannot check rank for missing topic '{topic_id}'. Got {shortlist_ids}",
            )
            self.assertLessEqual(
                rank_by_topic[topic_id],
                max_rank,
                msg=(
                    f"{case.name}: expected '{topic_id}' within top {max_rank} for query {case.query!r}. "
                    f"Got rank {rank_by_topic[topic_id]} and shortlist {shortlist_ids}"
                ),
            )


def _build_test(case: ShortlistCase):
    def test_method(self: TopicShortlistBuilderTests) -> None:
        self._assert_case(case)

    test_method.__name__ = f"test_{case.name}"
    return test_method


for _case in SHORTLIST_CASES:
    setattr(TopicShortlistBuilderTests, f"test_{_case.name}", _build_test(_case))


if __name__ == "__main__":
    unittest.main()
