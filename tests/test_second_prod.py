from __future__ import annotations

import logging
import re
import sys
import unittest
from pathlib import Path
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.main import build_app


SEMANTIC_EQUIVALENTS = {
    "pricing_epc_tis": ["epc", "тис", "tis", "епс", "тариф", "стоим", "руб"],
    "catalog_list_detailed": ["бренд", "марк", "каталог", "mercedes", "toyota", "volvo", "bmw", "audi", "honda"],
    "services_advantages": ["преимущ", "актуаль", "точн", "стабил", "vin", "данн"],
    "compare_epc_tis": ["epc", "tis", "епс", "тис", "отлич", "разниц"],
    "post_payment_access": ["после оплат", "доступ", "подключ", "активац"],
    "refund_policy": ["возврат", "деньг", "услов", "гарант"],
    "manager_not_loop": ["менеджер", "номер", "связ"],
    "nonsense_clarify": ["уточните", "что именно", "цены", "бренды", "оформлен"],
    "demo_policy": ["демо", "доступ", "юрид", "ип"],
    "multi_user": ["нескольк", "пользоват", "доступ", "количество"],
    "brand_available": ["да", "каталог", "бренд", "марк"],
    "out_of_scope": ["не наш профиль", "не подходит", "не подойдет", "не про автокаталоги"],
    "human_deescalation": ["помогу", "уточните", "подскажу"],
}

PRICE_TOPICS = {"epc_tariffs", "tis_tariffs"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _expand_needles(needles: list[str]) -> list[str]:
    expanded: list[str] = []
    for needle in needles:
        key = _normalize(needle)
        if key in SEMANTIC_EQUIVALENTS:
            expanded.extend(_normalize(item) for item in SEMANTIC_EQUIVALENTS[key])
        else:
            expanded.append(key)
    result: list[str] = []
    for item in expanded:
        if item and item not in result:
            result.append(item)
    return result


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in _expand_needles(needles))


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(n in text for n in _expand_needles(needles))


def _allowed_evidence_sources(topic_ids: list[str]) -> set[str]:
    topics = {str(topic_id).strip() for topic_id in topic_ids if str(topic_id).strip()}
    allowed = {"facts.yaml"}
    if topics & PRICE_TOPICS:
        allowed.add("prices.yaml")
    return allowed


def _make_turn(
    user: str,
    expected_topics_any: list[str] | None = None,
    must_include_any: list[str] | None = None,
    must_include_all: list[str] | None = None,
    must_not_include: list[str] | None = None,
    expected_actions_any: list[str] | None = None,
    must_differ_from_prev_bot: bool = False,
) -> dict[str, object]:
    return {
        "user": user,
        "expected_topics_any": expected_topics_any or [],
        "must_include_any": must_include_any or [],
        "must_include_all": must_include_all or [],
        "must_not_include": must_not_include or [],
        "expected_actions_any": expected_actions_any or [],
        "must_differ_from_prev_bot": must_differ_from_prev_bot,
    }


SCENARIOS: list[dict[str, object]] = [
    {"name": "01_price_subscription_should_include_tariffs", "turns": [
        _make_turn("сколько стоит подписка", ["epc_tariffs", "tis_tariffs"], ["pricing_epc_tis"])
    ]},
    {"name": "02_catalog_list_should_not_be_too_generic", "turns": [
        _make_turn("какие каталоги у вас есть", ["brand_list_request", "specific_brand_check", "company_services_info"], ["catalog_list_detailed"])
    ]},
    {"name": "03_advantages_subscription", "turns": [
        _make_turn("какие преимущества вашей подписки", ["company_services_info", "price_objection"], ["services_advantages"])
    ]},
    {"name": "04_catalog_list_repeat_first", "turns": [
        _make_turn("какие каталоги у вас есть", ["brand_list_request", "company_services_info"], ["catalog_list_detailed"])
    ]},
    {"name": "05_catalog_list_repeat_second", "turns": [
        _make_turn("какие каталоги у вас есть", ["brand_list_request", "company_services_info"], ["catalog_list_detailed"])
    ]},
    {"name": "06_compare_catalogs", "turns": [
        _make_turn("чем отличается один каталог от другого", ["free_catalog_comparison", "epc_tariffs", "tis_tariffs"], ["compare_epc_tis"])
    ]},
    {"name": "07_which_catalog_mercedes", "turns": [
        _make_turn("какой каталог мне нужен для мерседес", ["specific_brand_check", "brand_list_request"], ["brand_available"])
    ]},
    {"name": "08_post_payment_access", "turns": [
        _make_turn("после оплаты доступ сразу появиться", ["post_payment_access_timing", "payment_without_details", "legal_entity_purchase_flow"], ["post_payment_access"])
    ]},
    {"name": "09_refund_policy", "turns": [
        _make_turn("можно вернуть деньги если не подошло", ["post_payment_no_access", "company_services_info"], ["refund_policy", "post_payment_access"])
    ]},
    {"name": "10_catalog_followup_with_brand", "turns": [
        _make_turn("а какие есть", ["brand_list_request", "company_services_info"], ["catalog_list_detailed"]),
        _make_turn("мерседес", ["specific_brand_check", "brand_list_request"], ["brand_available"]),
    ]},
    {"name": "11_ambiguous_price_question", "turns": [
        _make_turn("сколько стоит", ["epc_tariffs", "tis_tariffs"], ["pricing_epc_tis"])
    ]},
    {"name": "12_abuse_should_not_trigger_checkout", "turns": [
        _make_turn(
            "ты тупишь",
            ["nonsense_input", "human_operator_request"],
            ["human_deescalation", "nonsense_clarify"],
            must_not_include=["инн", "телефон", "qr", "счет"],
        )
    ]},
    {"name": "13_noise_in_price_request", "turns": [
        _make_turn("пришли информацию на год епс верхние лобки", ["epc_tariffs"], ["pricing_epc_tis"])
    ]},
    {"name": "14_unknown_brand_global_hawk", "turns": [
        _make_turn("каталог на global hawk есть? мой сбили хочу отремонтировать", ["out_of_scope_request", "specific_brand_check"], ["out_of_scope", "brand_available"])
    ]},
    {"name": "15_buy_and_price_combo", "turns": [
        _make_turn("хочу доступ к каталогам что по деньгам", ["purchase_ready", "epc_tariffs", "tis_tariffs"], ["pricing_epc_tis"])
    ]},
    {"name": "16_dot_nonsense", "turns": [
        _make_turn(
            ".",
            ["nonsense_input", "out_of_scope_request"],
            ["nonsense_clarify"],
            must_not_include=["инн", "телефон", "qr", "счет"],
            expected_actions_any=["clarify_request", "out_of_scope_response"],
        )
    ]},
    {"name": "17_random_nonsense", "turns": [
        _make_turn(
            "мммммммаааааа 235",
            ["nonsense_input", "out_of_scope_request"],
            ["nonsense_clarify"],
            must_not_include=["инн", "телефон", "qr", "счет"],
            expected_actions_any=["clarify_request", "out_of_scope_response"],
        )
    ]},
    {"name": "18_multi_intent_long_query", "turns": [
        _make_turn(
            "здравствуйте я хочу разобраться какие каталоги у вас доступны сколько стоит подписка какие есть тарифы, можно ли оплатить по счёту или qr коду, можно ли подключить несколько сотрудников, есть ли тестовый период, чем отличаются епс от тис",
            ["company_services_info", "epc_tariffs", "tis_tariffs", "free_catalog_comparison", "demo_access"],
            ["catalog_list_detailed", "pricing_epc_tis", "compare_epc_tis", "multi_user"],
        )
    ]},
    {"name": "19_abuse_message_no_checkout", "turns": [
        _make_turn(
            "вы мошенники",
            ["nonsense_input", "human_operator_request"],
            ["human_deescalation", "nonsense_clarify"],
            must_not_include=["инн", "телефон", "qr", "счет"],
        )
    ]},
    {"name": "20_free_access_request", "turns": [
        _make_turn("дай бесплатный доступ", ["demo_access"], ["demo_policy"])
    ]},
    {"name": "21_free_forever_claim", "turns": [
        _make_turn("мне обещали бесплатно навсегда", ["demo_access"], ["demo_policy"])
    ]},
    {"name": "22_multi_intent_price_buy_login", "turns": [
        _make_turn("сколько стоит мерседес, как купить и почему не работает вход", ["epc_tariffs", "tis_tariffs", "purchase_ready", "company_services_info"], ["pricing_epc_tis"])
    ]},
    {"name": "23_subscription_price_catalog", "turns": [
        _make_turn("сколько стоит подписка на каталог", ["epc_tariffs", "tis_tariffs"], ["pricing_epc_tis"])
    ]},
    {"name": "24_service_cons_question", "turns": [
        _make_turn("какие минусы у сервиса", ["company_services_info", "price_objection"], ["services_advantages", "human_deescalation"])
    ]},
    {"name": "25_short_catalog_list", "turns": [
        _make_turn("ответь коротко какие каталоги есть", ["brand_list_request", "company_services_info"], ["catalog_list_detailed"])
    ]},
    {"name": "26_detailed_subscription_info", "turns": [
        _make_turn("расскажи подробно про подписку", ["epc_tariffs", "tis_tariffs", "company_services_info"], ["pricing_epc_tis"])
    ]},
    {"name": "27_cheaper_than_tis_contextual", "turns": [
        _make_turn("расскажи подробно про подписку", ["epc_tariffs", "tis_tariffs"], ["pricing_epc_tis"]),
        _make_turn("а это дешевле чем тис", ["free_catalog_comparison", "tis_tariffs", "epc_tariffs"], ["compare_epc_tis"], must_differ_from_prev_bot=True),
    ]},
    {"name": "28_multi_user_access", "turns": [
        _make_turn("можно ли пользоваться нескольким людям", ["multi_device_access"], ["multi_user"])
    ]},
    {"name": "29_catalog_loop_prevention", "turns": [
        _make_turn("какие есть каталоги", ["brand_list_request", "company_services_info"], ["catalog_list_detailed"]),
        _make_turn("какие марки есть каталоге", ["brand_list_request", "company_services_info"], ["catalog_list_detailed"], must_differ_from_prev_bot=True),
        _make_turn("каталоги каких марок есть", ["brand_list_request", "company_services_info"], ["catalog_list_detailed"], must_differ_from_prev_bot=True),
    ]},
    {"name": "30_brand_volvo_and_uaz", "turns": [
        _make_turn("есть ли каталог вольво", ["specific_brand_check", "brand_list_request"], ["brand_available"]),
        _make_turn("есть ли каталог уаз", ["specific_brand_check", "brand_list_request"], ["brand_available"]),
    ]},
]


class SecondProdDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.tests_log_path = cls.project_root / "logs/tests.log"
        cls.logger = logging.getLogger("second_prod_tests")
        cls.logger.setLevel(logging.INFO)
        cls.logger.handlers.clear()
        cls.logger.propagate = False
        handler = logging.FileHandler(cls.tests_log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        cls.logger.addHandler(handler)
        cls.logger.info("===== SECOND PROD TEST RUN START =====")
        cls.app = build_app(cls.project_root)
        policy_path = cls.project_root / "src/config/response_policy.yaml"
        policy_raw = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
        templates_raw = policy_raw.get("templates", {}) if isinstance(policy_raw, dict) else {}
        cls._template_phrases: set[str] = set()
        if isinstance(templates_raw, dict):
            for variants in templates_raw.values():
                if not isinstance(variants, list):
                    continue
                for item in variants:
                    text = _normalize(str(item))
                    if text:
                        cls._template_phrases.add(text)
        cls._service_actions = {"clarify_request", "out_of_scope_response", "closing_ack", "greeting_once"}
        if len(SCENARIOS) != 30:
            raise AssertionError(f"Expected 30 scenarios, got {len(SCENARIOS)}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.info("===== SECOND PROD TEST RUN END =====")
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
        answer = _normalize(response.answer_text)

        expected_topics_any = [str(item) for item in turn.get("expected_topics_any", [])]
        expected_actions_any = [str(item) for item in turn.get("expected_actions_any", [])]
        must_include_any = [str(item) for item in turn.get("must_include_any", [])]
        must_include_all = [str(item) for item in turn.get("must_include_all", [])]
        must_not_include = [str(item) for item in turn.get("must_not_include", [])]
        must_differ = bool(turn.get("must_differ_from_prev_bot", False))

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
                "must_include_all": must_include_all,
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
                f"[{scenario_name}][turn {turn_index}] topic mismatch. expected any={expected_topics_any}, got={response.topic_ids}"
            )
        if expected_actions_any and response.action_name not in expected_actions_any:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] action mismatch. expected any={expected_actions_any}, got={response.action_name}"
            )
        if must_include_any and not _contains_any(answer, must_include_any):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_include_any failed. needles={must_include_any}, answer={response.answer_text}"
            )
        if must_include_all and not _contains_all(answer, must_include_all):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_include_all failed. needles={must_include_all}, answer={response.answer_text}"
            )
        if must_not_include and _contains_any(answer, must_not_include):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_not_include violated. needles={must_not_include}, answer={response.answer_text}"
            )
        if must_differ and prev_bot_answer and answer == _normalize(prev_bot_answer):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] repeated_bot_answer. answer={response.answer_text}"
            )
        if response.action_name != "clarify_request" and not response.answer_sections:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] answer_sections is empty for non-clarify action."
            )
        if response.action_name != "clarify_request" and not response.used_evidence_ids:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] used_evidence_ids is empty for non-clarify action."
            )
        if response.action_name != "clarify_request":
            evidence_items = response.evidence_pack.items if response.evidence_pack else []
            if not evidence_items:
                response_ok = False
                failures.append(
                    f"[{scenario_name}][turn {turn_index}] evidence_pack.items is empty for non-clarify action."
                )
            else:
                missing_ids = [item.evidence_id for item in evidence_items if not str(item.evidence_id).strip()]
                if missing_ids:
                    response_ok = False
                    failures.append(
                        f"[{scenario_name}][turn {turn_index}] evidence item without evidence_id."
                    )
                allowed_sources = _allowed_evidence_sources(response.topic_ids)
                weak_sources = [
                    str(item.source)
                    for item in evidence_items
                    if str(item.source).strip() not in allowed_sources
                ]
                if weak_sources:
                    response_ok = False
                    failures.append(
                        f"[{scenario_name}][turn {turn_index}] evidence source not allowed: {weak_sources[:2]} | allowed={sorted(allowed_sources)}"
                    )
        if response.action_name not in self._service_actions:
            if answer in self._template_phrases:
                response_ok = False
                failures.append(
                    f"[{scenario_name}][turn {turn_index}] template leakage: answer equals policy template for non-service action."
                )
        if response.action_name != "clarify_request" and not response.contract_flags.get("planned_action_matches", False):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] contract invariant failed: planned_action_matches=False."
            )
        if response.action_name != "clarify_request" and not response.contract_flags.get("trace_complete", False):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] trace invariant failed: trace_complete=False."
            )
        if response.action_name != "clarify_request" and "evidence=" not in response.reasoning_summary:
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] reasoning invariant failed: evidence marker missing."
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
    def test_method(self: SecondProdDialogTests) -> None:
        session_id = f"second-prod-{scenario['name']}"
        self.app.clear_session(session_id)
        prev_bot_answer: str | None = None
        for idx, turn in enumerate(scenario["turns"], start=1):
            prev_bot_answer = self._run_turn(
                session_id=session_id,
                scenario_name=str(scenario["name"]),
                turn_index=idx,
                turn=turn,
                prev_bot_answer=prev_bot_answer,
            )
        self.app.clear_session(session_id)

    return test_method


for _scenario in SCENARIOS:
    setattr(
        SecondProdDialogTests,
        f"test_second_prod_{_scenario['name']}",
        _build_test_case(_scenario),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

# cd /root/project/Chat_bot && CHATBOT_API_KEY="xK9mLpQ2vN7wR" /root/project/.venv/bin/python tests/test_second_prod.py
