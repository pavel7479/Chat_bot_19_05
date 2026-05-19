from __future__ import annotations

import logging
import re
import sys
import unittest
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.main import build_app


SEMANTIC_EQUIVALENTS = {
    "services_help": [
        "каталог",
        "epc",
        "tis",
        "что именно нужно",
        "могу подсказать",
        "помогу",
    ],
    "company_services": [
        "каталог",
        "автосервис",
        "автобизнес",
        "профессиональ",
        "услуг",
    ],
    "brand_available": [
        "да, у нас есть",
        "есть каталоги",
        "шахман",
        "shacman",
        "уточните интересующую марку",
    ],
    "clarify_not_checkout": [
        "уточните",
        "что именно нужно",
        "не оформляю",
        "спокойно",
        "работаем с каталогами автозапчастей",
    ],
    "not_for_scope": [
        "не подходит",
        "не подойдет",
        "скорее всего наш сервис вам не подойдет",
        "не про автокаталоги",
        "автокаталоги",
        "работаем с каталогами автозапчастей",
        "если будет вопрос по автокаталогам",
    ],
    "no_requisites_loop": [
        "инн",
        "телефон",
        "счет",
        "qr",
    ],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _expand_needles(needles: list[str]) -> list[str]:
    expanded: list[str] = []
    for needle in needles:
        key = _normalize_text(needle)
        if key in SEMANTIC_EQUIVALENTS:
            expanded.extend(_normalize_text(item) for item in SEMANTIC_EQUIVALENTS[key])
        else:
            expanded.append(key)
    deduped: list[str] = []
    for item in expanded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _contains_any(text: str, needles: list[str]) -> bool:
    expanded = _expand_needles(needles)
    return any(needle in text for needle in expanded)


def _has_greeting_prefix(text: str) -> bool:
    return bool(re.match(r"^(здравствуйте|добрый день|добрый вечер|привет)\b", text))


def _make_turn(
    user: str,
    expected_topics_any: list[str] | None = None,
    must_include_any: list[str] | None = None,
    must_not_include: list[str] | None = None,
    must_differ_from_prev_bot: bool = False,
    expected_actions_any: list[str] | None = None,
) -> dict[str, object]:
    return {
        "user": user,
        "expected_topics_any": expected_topics_any or [],
        "expected_actions_any": expected_actions_any or [],
        "must_include_any": must_include_any or [],
        "must_not_include": must_not_include or [],
        "must_differ_from_prev_bot": must_differ_from_prev_bot,
    }


SCENARIOS: list[dict[str, object]] = [
    {
        "name": "01_help_flow_basic",
        "turns": [
            _make_turn("Чем ты мне можешь помочь?", ["company_services_info"], ["services_help"]),
            _make_turn("подскажи", ["company_services_info"], ["services_help"], ["no_requisites_loop"]),
        ],
    },
    {
        "name": "02_competitor_value_discounts",
        "turns": [
            _make_turn(
                "чем вы лучше ваших конкурентов. можете ли заинтересовать меня как клиента. Есть ли у вас скидки?",
                ["company_services_info", "price_objection", "competitor_choice"],
                ["company_services"],
            ),
            _make_turn("подскажи коротко", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "03_docs_request_no_checkout",
        "turns": [
            _make_turn(
                "хочу ознакомиться с договором и получить карточку вашего предприятия",
                ["company_services_info", "human_operator_request"],
                ["company_services", "clarify_not_checkout"],
                ["no_requisites_loop"],
            ),
            _make_turn(
                "повтори, какие документы доступны",
                ["company_services_info", "human_operator_request"],
                ["company_services", "clarify_not_checkout"],
                ["no_requisites_loop"],
            ),
        ],
    },
    {
        "name": "04_docs_then_contacts",
        "turns": [
            _make_turn(
                "хочу ознакомиться с договором и получить карточку вашего предприятия",
                ["company_services_info", "human_operator_request"],
                ["company_services", "clarify_not_checkout"],
            ),
            _make_turn(
                "ИНН 2320140650, +79990875689 Иван",
                ["company_services_info", "human_operator_request"],
                ["clarify_not_checkout", "company_services"],
            ),
        ],
    },
    {
        "name": "05_shacman_brand_check",
        "turns": [
            _make_turn("нужен каталог шахман", ["specific_brand_check"], ["brand_available"]),
            _make_turn("а для shacman тоже есть?", ["specific_brand_check"], ["brand_available"]),
        ],
    },
    {
        "name": "06_capabilities_dialog",
        "turns": [
            _make_turn("что ты умеешь", ["company_services_info"], ["services_help"]),
            _make_turn("подскажи", ["company_services_info"], ["services_help"], ["no_requisites_loop"]),
            _make_turn("какой каталог лучше использовать", ["company_services_info"], ["services_help"]),
            _make_turn("подбор деталей выполняется через ваши каталоги?", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "07_deescalation_where_forming",
        "turns": [
            _make_turn("Хочу купить доступ", ["purchase_ready", "legal_status_unknown"]),
            _make_turn("это куда ты меня оформлять собрался", ["company_services_info"], ["clarify_not_checkout"], ["no_requisites_loop"]),
        ],
    },
    {
        "name": "08_deescalation_hot_head",
        "turns": [
            _make_turn("Хочу купить доступ", ["purchase_ready", "legal_status_unknown"]),
            _make_turn("ты горячку не пори, а то чуть что сразу оформлять", ["company_services_info"], ["clarify_not_checkout"], ["no_requisites_loop"]),
        ],
    },
    {
        "name": "09_deescalation_out_of_scope_business",
        "turns": [
            _make_turn("Хочу купить доступ", ["purchase_ready", "legal_status_unknown"]),
            _make_turn("мы шьём одежд для редких пород рыб", ["out_of_scope_request"], ["not_for_scope"], ["no_requisites_loop"]),
        ],
    },
    {
        "name": "10_help_short_followup",
        "turns": [
            _make_turn("чем ты мне можешь помочь?", ["company_services_info"], ["services_help"]),
            _make_turn("подскажи пожалуйста", ["company_services_info"], ["services_help"], ["no_requisites_loop"]),
        ],
    },
    {
        "name": "11_best_catalog_no_checkout",
        "turns": [
            _make_turn("какой каталог лучше использовать", ["company_services_info"], ["services_help"]),
            _make_turn("в двух словах", ["company_services_info"], ["services_help"], ["no_requisites_loop"]),
        ],
    },
    {
        "name": "12_parts_catalog_confirmation",
        "turns": [
            _make_turn("подбор деталей выполняется через ваши каталоги?", ["company_services_info"], ["services_help"]),
            _make_turn("то есть чат-бот не подбирает деталь сам?", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "13_competitor_then_services",
        "turns": [
            _make_turn("чем вы лучше конкурентов?", ["company_services_info", "competitor_choice"], ["company_services"]),
            _make_turn("подскажи, что умеет система", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "14_docs_then_help",
        "turns": [
            _make_turn("нужен договор и карточка компании", ["company_services_info", "human_operator_request"], ["company_services", "clarify_not_checkout"]),
            _make_turn("чем еще можете помочь?", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "15_brand_then_services",
        "turns": [
            _make_turn("нужен каталог шахман", ["specific_brand_check"], ["brand_available"]),
            _make_turn("что еще умеет бот?", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "16_help_then_price_question",
        "turns": [
            _make_turn("что ты умеешь", ["company_services_info"], ["services_help"]),
            _make_turn("сколько стоит epc full?", ["epc_tariffs"], ["5500"]),
        ],
    },
    {
        "name": "17_services_then_buy_intent_explicit",
        "turns": [
            _make_turn("какой каталог лучше использовать", ["company_services_info"], ["services_help"]),
            _make_turn("ок, теперь как купить доступ для ип", ["purchase_ready", "legal_entity_purchase_flow"], ["инн"]),
        ],
    },
    {
        "name": "18_out_of_scope_after_services",
        "turns": [
            _make_turn("что ты умеешь", ["company_services_info"], ["services_help"]),
            _make_turn("мы шьём одежд для редких пород рыб", ["out_of_scope_request"], ["not_for_scope"]),
        ],
    },
    {
        "name": "19_emotional_then_clarify",
        "turns": [
            _make_turn("это куда ты меня оформлять собрался", ["company_services_info"], ["clarify_not_checkout"]),
            _make_turn("ладно, какой каталог лучше использовать", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "20_dialog_full_reference",
        "turns": [
            _make_turn("что ты умеешь", ["company_services_info"], ["services_help"]),
            _make_turn("подскажи", ["company_services_info"], ["services_help"], ["no_requisites_loop"], must_differ_from_prev_bot=True),
            _make_turn("какой каталог лучше использовать", ["company_services_info"], ["services_help"], must_differ_from_prev_bot=True),
            _make_turn("подбор деталей выполняется через ваши каталоги?", ["company_services_info"], ["services_help"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "21_user_case_help_then_hint",
        "turns": [
            _make_turn("Чем ты мне можешь помочь?", ["company_services_info"], ["services_help"]),
            _make_turn("подскажи", ["company_services_info"], ["services_help"], ["no_requisites_loop"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "22_user_case_competitor_discount",
        "turns": [
            _make_turn(
                "чем вы лучше ваших конкурентов. можете ли заинтересовать меня как клиента. Есть ли у вас скидки?",
                ["company_services_info", "competitor_choice", "price_objection"],
                ["company_services"],
            ),
        ],
    },
    {
        "name": "23_user_case_docs_then_contacts",
        "turns": [
            _make_turn(
                "хочу ознакомиться с договором и получить карточку вашего предприятия",
                ["company_services_info", "human_operator_request"],
                ["company_services", "clarify_not_checkout"],
                ["no_requisites_loop"],
            ),
            _make_turn(
                "ИНН 2320140650, +79990875689 Иван",
                ["company_services_info", "human_operator_request"],
                ["clarify_not_checkout", "company_services"],
                ["no_requisites_loop"],
            ),
        ],
    },
    {
        "name": "24_user_case_shacman",
        "turns": [
            _make_turn("нужен каталог шахман", ["specific_brand_check"], ["brand_available"]),
        ],
    },
    {
        "name": "25_user_case_capability_chain",
        "turns": [
            _make_turn("что ты умеешь", ["company_services_info"], ["services_help"]),
            _make_turn("подскажи", ["company_services_info"], ["services_help"], ["no_requisites_loop"], must_differ_from_prev_bot=True),
            _make_turn("какой каталог лучше использовать", ["company_services_info"], ["services_help"], must_differ_from_prev_bot=True),
            _make_turn("подбор деталей выполняется через ваши каталоги?", ["company_services_info"], ["services_help"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "26_user_case_deescalation_forming",
        "turns": [
            _make_turn("это куда ты меня оформлять собрался", ["company_services_info"], ["clarify_not_checkout"], ["no_requisites_loop"]),
            _make_turn("ты горячку не пори, а то чуть что сразу оформлять", ["company_services_info"], ["clarify_not_checkout"], ["no_requisites_loop"]),
            _make_turn("мы шьем одежду для редких пород рыб", ["out_of_scope_request"], ["not_for_scope"]),
        ],
    },
    {
        "name": "27_topic_switch_api_then_buy",
        "turns": [
            _make_turn("у вас есть api?", ["api_integration"], ["api"]),
            _make_turn("тогда как оформить доступ на ип", ["purchase_ready", "legal_entity_purchase_flow"], ["инн"]),
        ],
    },
    {
        "name": "28_topic_switch_tis_after_epc_history",
        "turns": [
            _make_turn("сколько стоит epc full", ["epc_tariffs"], ["5500"]),
            _make_turn("а сколько tис для honda", ["tis_tariffs"], ["tis"]),
        ],
    },
    {
        "name": "29_topic_switch_epc_after_tis_history",
        "turns": [
            _make_turn("сколько tis для toyota", ["tis_tariffs"], ["tis"]),
            _make_turn("а epc full сколько стоит", ["epc_tariffs"], ["5500"]),
        ],
    },
    {
        "name": "30_service_signal_must_not_start_checkout",
        "turns": [
            _make_turn("чем можете помочь", ["company_services_info"], ["services_help"], ["no_requisites_loop"]),
            _make_turn("подскажи по возможностям", ["company_services_info"], ["services_help"], ["no_requisites_loop"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "31_docs_not_checkout_even_with_legal_data",
        "turns": [
            _make_turn("нужен договор и карточка компании", ["company_services_info", "human_operator_request"], ["clarify_not_checkout"], ["no_requisites_loop"]),
            _make_turn("мы ип, инн 7700000000", ["company_services_info", "human_operator_request"], ["clarify_not_checkout"], ["no_requisites_loop"]),
        ],
    },
    {
        "name": "32_out_of_scope_after_domain",
        "turns": [
            _make_turn("сколько стоит доступ", ["purchase_ready", "legal_status_unknown", "epc_tariffs", "tis_tariffs"]),
            _make_turn(
                "а каталог велосипедов есть?",
                ["out_of_scope_request"],
                ["not_for_scope"],
                expected_actions_any=["out_of_scope_response"],
            ),
        ],
    },
    {
        "name": "33_out_of_scope_joke",
        "turns": [
            _make_turn(
                "нужен каталог для космического корабля",
                ["out_of_scope_request"],
                ["not_for_scope"],
                expected_actions_any=["out_of_scope_response"],
            ),
        ],
    },
    {
        "name": "34_parts_selection_request",
        "turns": [
            _make_turn("подберите деталь по vin", ["company_services_info"], ["services_help"]),
            _make_turn("то есть вы не подбираете напрямую?", ["company_services_info"], ["services_help"]),
        ],
    },
    {
        "name": "35_brand_vaz_lada_synonyms",
        "turns": [
            _make_turn("нужны каталоги ваза", ["specific_brand_check", "brand_list_request"], ["brand_available"]),
            _make_turn("а по lada есть?", ["specific_brand_check"], ["brand_available"]),
            _make_turn("и по автоваз?", ["specific_brand_check"], ["brand_available"]),
        ],
    },
    {
        "name": "36_compare_epc_tis_direct",
        "turns": [
            _make_turn(
                "в чем отличие тис от epc",
                ["free_catalog_comparison", "tis_tariffs", "epc_tariffs", "company_services_info"],
                ["epc", "tis"],
                expected_actions_any=["compare_epc_tis", "tis_tariffs", "epc_tariffs"],
            ),
        ],
    },
    {
        "name": "37_company_services_non_repeat_three_turns",
        "turns": [
            _make_turn("что умеет бот", ["company_services_info"], ["services_help"]),
            _make_turn("а подробнее", ["company_services_info"], ["services_help"], must_differ_from_prev_bot=True),
            _make_turn("еще короче", ["company_services_info"], ["services_help"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "38_manager_first_then_second_request",
        "turns": [
            _make_turn("позовите менеджера", ["human_operator_request"], ["менеджер"]),
            _make_turn("нет, именно менеджера подключите", ["human_operator_request"], ["номер", "менеджер"]),
        ],
    },
    {
        "name": "39_manager_phone_validation_flow",
        "turns": [
            _make_turn(
                "позовите менеджера",
                ["human_operator_request"],
                ["менеджер"],
                expected_actions_any=["human_operator", "human_operator_collect_phone"],
            ),
            _make_turn(
                "нужен звонок менеджера",
                ["human_operator_request"],
                ["номер", "менеджер"],
                expected_actions_any=["human_operator_collect_phone"],
            ),
            _make_turn(
                "мой номер 12345",
                ["human_operator_request"],
                ["11", "цифр"],
                expected_actions_any=["human_operator_phone_invalid"],
            ),
            _make_turn(
                "+79991234567",
                ["human_operator_request"],
                ["менеджер", "свяжется"],
                expected_actions_any=["human_operator_phone_confirm"],
            ),
        ],
    },
    {
        "name": "40_chain_regression_services_docs_discounts",
        "turns": [
            _make_turn("чем ты мне можешь помочь?", ["company_services_info"], ["services_help"]),
            _make_turn("подскажи", ["company_services_info"], ["services_help"], ["no_requisites_loop"], must_differ_from_prev_bot=True),
            _make_turn(
                "чем вы лучше ваших конкурентов. можете ли заинтересовать меня как клиента. Есть ли у вас скидки?",
                ["company_services_info", "price_objection", "competitor_choice"],
                ["company_services"],
            ),
            _make_turn(
                "хочу ознакомиться с договором и получить карточку вашего предприятия",
                ["company_services_info", "human_operator_request"],
                ["company_services", "clarify_not_checkout"],
                ["no_requisites_loop"],
                must_differ_from_prev_bot=True,
            ),
        ],
    },
]


class FirstProdDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.tests_log_path = cls.project_root / "logs/tests.log"
        cls.logger = logging.getLogger("first_prod_tests")
        cls.logger.setLevel(logging.INFO)
        cls.logger.handlers.clear()
        cls.logger.propagate = False
        handler = logging.FileHandler(cls.tests_log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        cls.logger.addHandler(handler)
        cls.logger.info("===== FIRST PROD TEST RUN START =====")

        cls.app = build_app(cls.project_root)
        if len(SCENARIOS) != 40:
            raise AssertionError(f"Expected 40 scenarios, got {len(SCENARIOS)}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.info("===== FIRST PROD TEST RUN END =====")
        for handler in cls.logger.handlers:
            handler.close()
        cls.logger.handlers.clear()

    def _run_turn(
        self,
        session_id: str,
        scenario_name: str,
        turn_index: int,
        turn: dict[str, object],
        prev_bot_answer: str | None = None,
    ) -> str:
        user_query = str(turn["user"])
        response = self.app.respond(session_id=session_id, user_query=user_query)
        answer_text = _normalize_text(response.answer_text)

        expected_topics_any = [str(item) for item in turn.get("expected_topics_any", [])]
        expected_actions_any = [str(item) for item in turn.get("expected_actions_any", [])]
        must_include_any = [str(item) for item in turn.get("must_include_any", [])]
        must_not_include = [str(item) for item in turn.get("must_not_include", [])]
        must_differ_from_prev_bot = bool(turn.get("must_differ_from_prev_bot", False))

        self.logger.info(
            "scenario=%s | turn=%s | precheck query=%s | topics=%s | planned_action=%s | final_action=%s | checks=%s | answer=%s",
            scenario_name,
            turn_index,
            user_query,
            response.topic_ids,
            response.planned_action,
            response.action_name,
            {
                "expected_topics_any": expected_topics_any,
                "expected_actions_any": expected_actions_any,
                "must_include_any": must_include_any,
                "must_not_include": must_not_include,
            },
            response.answer_text,
        )

        failures: list[str] = []
        classifier_ok = True
        response_ok = True

        if expected_topics_any and not any(topic in response.topic_ids for topic in expected_topics_any):
            classifier_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] topic mismatch. "
                f"expected any={expected_topics_any}, got={response.topic_ids}"
            )
        if expected_actions_any and response.action_name not in expected_actions_any:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] action mismatch. "
                f"expected any={expected_actions_any}, got={response.action_name}"
            )
        if not response.planned_action:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] planned_action is empty."
            )
        if response.action_name != "clarify_request" and not response.answer_sections:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] answer_sections is empty for non-clarify action."
            )
        if (
            response.planned_action
            and response.action_name != response.planned_action
            and response.action_name != "clarify_request"
        ):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] action invariant failed. "
                f"planned_action={response.planned_action}, final_action={response.action_name}"
            )
        if response.action_name != "clarify_request" and response.answer_sections and not response.used_evidence_ids:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] evidence invariant failed. "
                "answer_sections present but used_evidence_ids is empty."
            )
        if response.action_name != "clarify_request" and not response.contract_flags.get("planned_action_matches", False):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] contract invariant failed. "
                "contract_flags.planned_action_matches is False."
            )
        if response.action_name != "clarify_request" and not response.contract_flags.get("trace_complete", False):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] trace invariant failed. "
                "contract_flags.trace_complete is False."
            )
        if response.action_name != "clarify_request" and "evidence=" not in response.reasoning_summary:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] reasoning invariant failed. "
                "reasoning_summary has no evidence marker."
            )
        if must_include_any and not _contains_any(answer_text, must_include_any):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_include_any failed. "
                f"needles={must_include_any}, answer={response.answer_text}"
            )
        if must_not_include and _contains_any(answer_text, must_not_include):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_not_include violated. "
                f"needles={must_not_include}, answer={response.answer_text}"
            )
        if must_differ_from_prev_bot and prev_bot_answer and answer_text == _normalize_text(prev_bot_answer):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] repeated_bot_answer. "
                f"answer={response.answer_text}"
            )
        if turn_index > 1 and _has_greeting_prefix(answer_text):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] repeated_greeting. "
                f"answer={response.answer_text}"
            )

        self.logger.info(
            "scenario=%s | turn=%s | postcheck query=%s | topics=%s | classifier_ok=%s | response_ok=%s | failed_rules=%s | answer=%s",
            scenario_name,
            turn_index,
            user_query,
            response.topic_ids,
            classifier_ok,
            response_ok,
            failures,
            response.answer_text,
        )
        if failures:
            self.fail("\n".join(failures))
        return response.answer_text


def _build_test_case(scenario: dict[str, object]):
    def test_method(self: FirstProdDialogTests) -> None:
        session_id = f"first-prod-{scenario['name']}"
        self.app.clear_session(session_id)
        prev_bot_answer: str | None = None
        for idx, turn in enumerate(scenario["turns"], start=1):
            prev_bot_answer = self._run_turn(
                session_id,
                str(scenario["name"]),
                idx,
                turn,
                prev_bot_answer=prev_bot_answer,
            )
        self.app.clear_session(session_id)

    return test_method


for _scenario in SCENARIOS:
    setattr(
        FirstProdDialogTests,
        f"test_first_prod_{_scenario['name']}",
        _build_test_case(_scenario),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

# cd /root/project/Chat_bot && CHATBOT_API_KEY="xK9mLpQ2vN7wR" /root/project/.venv/bin/python tests/test_first_prod.py
