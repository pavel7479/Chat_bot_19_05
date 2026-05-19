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
    "pricing_epc": ["epc", "епс", "тариф", "стоим", "руб", "6500", "18000", "34800", "62400"],
    "pricing_tis_brand_required": ["tis", "тис", "бренд", "марк", "стоим"],
    "pricing_tis_mercedes": ["tis", "10000", "mercedes", "мерседес"],
    "pricing_tis_volvo": ["tis", "10000", "volvo", "вольво"],
    "catalog_examples": ["bmw", "mercedes", "toyota", "volvo", "zeekr", "lada", "уаз"],
    "catalog_presence": ["каталог", "бренд", "марк", "доступ"],
    "service_help": ["каталог", "возможност", "подскажу", "сервис"],
    "service_value": ["актуаль", "стабил", "vin", "скид", "спецуслов"],
    "docs_info": ["договор", "карточ", "документ", "предприят"],
    "compare_epc_tis": ["epc", "tis", "отлич", "разниц", "отдельн", "бренд"],
    "multi_user": ["нескольк", "сотрудник", "доступ", "пользоват"],
    "demo_policy": ["демо", "юрид", "ип", "статус"],
    "nonsense_clarify": ["уточните", "что именно", "epc", "tis", "бренды", "оформлен"],
    "deescalation": ["помогу", "подскажу", "уточните"],
    "post_payment": ["после оплат", "доступ", "активац", "сразу"],
    "refund_support": ["доступ", "оплат", "номер", "менеджер", "свяж"],
    "out_of_scope": ["не подойдет", "не наш профиль", "не про автокаталоги"],
    "legal_not_physical": ["юрид", "ип", "компан"],
    "greeting_help": ["здравствуйте", "помогу", "подскажу"],
}

PRICE_TOPICS = {"epc_tariffs", "tis_tariffs"}
CHECKOUT_LEAK_NEEDLES = [
    "инн",
    "qr",
    "по счету",
    "выставить счет",
    "контактного лица",
    "желаемый период",
    "количество доступов",
]
PHYSICAL_REJECT_NEEDLES = ["частным лицам", "физичес", "не продаем"]
PARTS_SELECTION_TEMPLATE = ["подбор деталей выполняется через наши каталоги", "чат-бот не подбирает запчасти напрямую"]
COMPARE_ONLY_TEMPLATE = ["не входят в epc", "отдельные технические базы", "сервисные модули"]


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
    expected_actions_any: list[str] | None = None,
    must_include_any: list[str] | None = None,
    must_include_all: list[str] | None = None,
    must_not_include: list[str] | None = None,
    must_differ_from_prev_bot: bool = False,
    min_lines: int | None = None,
    max_lines: int | None = None,
) -> dict[str, object]:
    return {
        "user": user,
        "expected_topics_any": expected_topics_any or [],
        "expected_actions_any": expected_actions_any or [],
        "must_include_any": must_include_any or [],
        "must_include_all": must_include_all or [],
        "must_not_include": must_not_include or [],
        "must_differ_from_prev_bot": must_differ_from_prev_bot,
        "min_lines": min_lines,
        "max_lines": max_lines,
    }


SCENARIOS: list[dict[str, object]] = [
    {
        "name": "01_help_flow_epc_two_years",
        "turns": [
            _make_turn("Чем ты мне можешь помочь?", ["company_services_info"], ["company_services"], ["service_help"]),
            _make_turn("Подскажи.", ["company_services_info", "nonsense_input"], None, ["service_help", "nonsense_clarify"], must_differ_from_prev_bot=True),
            _make_turn("EPC", ["epc_tariffs"], ["epc_tariffs"], ["pricing_epc"]),
            _make_turn("А на два года сколько?", ["epc_tariffs"], ["epc_tariffs"], ["pricing_epc", "nonsense_clarify"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "02_tis_all_brands_flow_no_checkout",
        "turns": [
            _make_turn("TIS", ["tis_tariffs"], ["tis_tariffs"], ["pricing_tis_brand_required"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("Все", ["tis_tariffs", "brand_list_request"], None, ["pricing_tis_brand_required", "catalog_examples"], must_not_include=CHECKOUT_LEAK_NEEDLES, must_differ_from_prev_bot=True),
            _make_turn("Сколько будет стоить?", ["tis_tariffs"], ["tis_tariffs"], ["pricing_tis_brand_required"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("Все бренды какие есть.", ["brand_list_request"], ["brand_availability"], ["catalog_examples"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("Все каталоги TIS", ["tis_tariffs"], ["tis_tariffs"], ["pricing_tis_brand_required"], must_not_include=CHECKOUT_LEAK_NEEDLES, must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "03_generic_catalog_request_no_checkout",
        "turns": [
            _make_turn("мне нужен каталог, какие у вас есть?", ["brand_list_request", "company_services_info"], ["brand_availability", "company_services"], ["catalog_examples"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("а по грузовым что есть?", ["brand_list_request"], ["brand_availability"], ["catalog_examples"], must_not_include=CHECKOUT_LEAK_NEEDLES, must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "04_municipal_enterprise_not_physical",
        "turns": [
            _make_turn("мы муниципальное предприятие горводоканал, нам нужны каталоги", ["company_services_info", "purchase_ready", "brand_list_request"], None, ["legal_not_physical", "catalog_examples", "service_help"], must_not_include=PHYSICAL_REJECT_NEEDLES),
            _make_turn("мы юрлицо", ["legal_entity_purchase_flow", "purchase_ready"], None, ["legal_not_physical"], must_not_include=PHYSICAL_REJECT_NEEDLES, must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "05_competitor_value_and_discounts",
        "turns": [
            _make_turn("чем вы лучше ваших конкурентов. можете ли заинтересовать меня как клиента. Есть ли у вас скидки?", ["company_services_info", "price_objection"], None, ["service_value"], must_not_include=PARTS_SELECTION_TEMPLATE, min_lines=1),
        ],
    },
    {
        "name": "06_documents_request_then_contact_data",
        "turns": [
            _make_turn("хочу ознакомиться с договором и получить карточку вашего предприятия", ["company_services_info"], ["company_services"], ["docs_info"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("ИНН 2320140650, +79990875689 Иван", ["company_services_info", "legal_entity_purchase_flow", "human_operator_request"], None, ["docs_info", "менеджер", "спасибо", "свяж"], must_differ_from_prev_bot=True, must_not_include=["контактного лица", "желаемый период", "количество доступов"]),
        ],
    },
    {
        "name": "07_shacman_alias_brand_check",
        "turns": [
            _make_turn("нужен каталог шахман", ["specific_brand_check"], ["brand_availability"], ["catalog_presence"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("сколько TIS на него?", ["tis_tariffs"], ["tis_tariffs"], ["pricing_tis_brand_required"], must_not_include=CHECKOUT_LEAK_NEEDLES, must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "08_capabilities_loop_prevention",
        "turns": [
            _make_turn("что ты умеешь", ["company_services_info"], ["company_services"], ["service_help"]),
            _make_turn("подскажи", ["company_services_info", "nonsense_input"], None, ["service_help", "nonsense_clarify"], must_differ_from_prev_bot=True),
            _make_turn("какой каталог лучше использовать", ["brand_list_request", "specific_brand_check", "company_services_info"], None, ["catalog_examples", "service_help", "nonsense_clarify"], must_differ_from_prev_bot=True),
            _make_turn("подбор деталей выполняется через ваши каталоги?", ["company_services_info"], ["company_services"], ["service_help"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "09_checkout_pushback_and_out_of_scope_company",
        "turns": [
            _make_turn("это куда ты меня оформлять собрался", ["nonsense_input", "company_services_info"], None, ["deescalation", "nonsense_clarify"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("ты горячку не пори, а то чуть что сразу оформлять", ["nonsense_input", "human_operator_request"], None, ["deescalation", "nonsense_clarify"], must_not_include=CHECKOUT_LEAK_NEEDLES, must_differ_from_prev_bot=True),
            _make_turn("мы шьём одежду для редких пород рыб", ["out_of_scope_request"], ["out_of_scope_response"], ["out_of_scope"], must_not_include=["11 цифр", "телефон"]),
        ],
    },
    {
        "name": "10_greeting_should_not_gate_legal",
        "turns": [
            _make_turn("привет", ["company_services_info", "nonsense_input"], None, ["greeting_help", "service_help"], must_not_include=["мы работаем только с юридическими лицами"]),
            _make_turn("какие у вас есть каталоги", ["brand_list_request"], ["brand_availability"], ["catalog_examples"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "11_subscription_price_should_cover_epc_and_tis",
        "turns": [
            _make_turn("сколько стоит подписка", ["epc_tariffs", "tis_tariffs"], None, ["pricing_epc", "pricing_tis_brand_required"], must_include_all=["pricing_epc", "тис"]),
        ],
    },
    {
        "name": "12_catalog_list_should_enumerate_examples",
        "turns": [
            _make_turn("какие каталоги у вас есть", ["brand_list_request"], ["brand_availability"], ["catalog_examples"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "13_advantages_should_use_value_facts",
        "turns": [
            _make_turn("какие преимущества вашей подписки", ["company_services_info"], ["company_services"], ["service_value"], must_not_include=PARTS_SELECTION_TEMPLATE),
        ],
    },
    {
        "name": "14_compare_catalogs_not_generic_services",
        "turns": [
            _make_turn("чем отличается один каталог от другого", ["free_catalog_comparison", "epc_tariffs", "tis_tariffs"], None, ["compare_epc_tis"], must_not_include=["помогаю с каталогами epc/tis"]),
        ],
    },
    {
        "name": "15_mercedes_catalog_then_tis_price",
        "turns": [
            _make_turn("какой каталог мне нужен для мерседес", ["specific_brand_check", "brand_list_request"], ["brand_availability"], ["catalog_presence"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("сколько TIS на него?", ["tis_tariffs"], ["tis_tariffs"], ["pricing_tis_mercedes"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "16_post_payment_access_not_checkout",
        "turns": [
            _make_turn("после оплаты доступ сразу появиться", ["post_payment_access_timing"], ["post_payment_access_info"], ["post_payment"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "17_refund_not_checkout",
        "turns": [
            _make_turn("можно вернуть деньги если не подошло", ["post_payment_no_access", "company_services_info"], ["post_payment_no_access_handoff", "company_services"], ["refund_support"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "18_catalog_followup_brand_richer_than_one_line",
        "turns": [
            _make_turn("а какие есть", ["brand_list_request"], ["brand_availability"], ["catalog_examples"]),
            _make_turn("мерседес", ["specific_brand_check"], ["brand_availability"], ["catalog_presence", "мерседес"], must_differ_from_prev_bot=True, min_lines=1),
        ],
    },
    {
        "name": "19_ambiguous_price_not_compare_only",
        "turns": [
            _make_turn("сколько стоит", ["epc_tariffs", "tis_tariffs"], None, ["pricing_epc", "pricing_tis_brand_required"], must_not_include=COMPARE_ONLY_TEMPLATE),
        ],
    },
    {
        "name": "20_abuse_deescalation_no_checkout",
        "turns": [
            _make_turn("ты тупишь", ["nonsense_input", "human_operator_request"], None, ["deescalation", "nonsense_clarify"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "21_noisy_epc_year_request",
        "turns": [
            _make_turn("пришли информацию на год епс верхние лобки", ["epc_tariffs"], ["epc_tariffs"], ["pricing_epc"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "22_unknown_brand_not_checkout",
        "turns": [
            _make_turn("каталог на global hawk есть? мой сбили хочу отремонтировать", ["specific_brand_check", "out_of_scope_request"], None, ["catalog_presence", "out_of_scope"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "23_buy_and_price_not_requisites_only",
        "turns": [
            _make_turn("хочу доступ к каталогам что по деньгам", ["purchase_ready", "epc_tariffs", "tis_tariffs"], None, ["pricing_epc", "pricing_tis_brand_required"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "24_dot_noise_no_checkout",
        "turns": [
            _make_turn(".", ["nonsense_input", "out_of_scope_request"], ["clarify_request", "out_of_scope_response"], ["nonsense_clarify"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "25_random_noise_no_checkout",
        "turns": [
            _make_turn("мммммммаааааа 235", ["nonsense_input", "out_of_scope_request"], ["clarify_request", "out_of_scope_response"], ["nonsense_clarify"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "26_long_multi_intent_not_tis_only",
        "turns": [
            _make_turn(
                "здравствуйте я хочу разобраться какие каталоги у вас доступны сколько стоит подписка какие есть тарифы, можно ли оплатить по счёту или qr коду, можно ли подключить несколько сотрудников, есть ли тестовый период, чем отличаются епс от тис",
                ["epc_tariffs", "tis_tariffs", "brand_list_request", "legal_entity_purchase_flow", "demo_access", "free_catalog_comparison", "company_services_info"],
                None,
                ["pricing_epc", "catalog_examples", "compare_epc_tis", "multi_user", "demo_policy"],
            ),
        ],
    },
    {
        "name": "27_free_access_demo_not_tis",
        "turns": [
            _make_turn("дай бесплатный доступ", ["demo_access"], ["demo_policy"], ["demo_policy"], must_not_include=["tis для", "стоимость зависит от бренда"]),
            _make_turn("мне обещали бесплатно навсегда", ["demo_access"], None, ["demo_policy"], must_differ_from_prev_bot=True),
        ],
    },
    {
        "name": "28_service_cons_not_clarify",
        "turns": [
            _make_turn("какие минусы у сервиса", ["company_services_info", "price_objection"], None, ["service_value", "service_help"], must_not_include=["что именно нужно", "скорее всего наш сервис вам не подойдет"]),
        ],
    },
    {
        "name": "29_multi_user_not_checkout",
        "turns": [
            _make_turn("можно ли пользоваться нескольким людям", ["company_services_info", "legal_entity_purchase_flow"], None, ["multi_user"], must_not_include=CHECKOUT_LEAK_NEEDLES),
        ],
    },
    {
        "name": "30_volvo_and_uaz_brand_flow",
        "turns": [
            _make_turn("есть ли каталог вольво", ["specific_brand_check"], ["brand_availability"], ["catalog_presence"], must_not_include=CHECKOUT_LEAK_NEEDLES),
            _make_turn("а сколько TIS на него", ["tis_tariffs"], ["tis_tariffs"], ["pricing_tis_volvo"], must_differ_from_prev_bot=True),
            _make_turn("есть ли каталог уаз", ["specific_brand_check"], ["brand_availability"], ["catalog_presence", "уаз"], must_not_include=["уточните интересующую марку"], must_differ_from_prev_bot=True),
        ],
    },
]


class SecondProdDialogTests02(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.tests_log_path = cls.project_root / "logs/tests_02.log"
        cls.logger = logging.getLogger("second_prod_tests_02")
        cls.logger.setLevel(logging.INFO)
        cls.logger.handlers.clear()
        cls.logger.propagate = False
        handler = logging.FileHandler(cls.tests_log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        cls.logger.addHandler(handler)
        cls.logger.info("===== SECOND PROD 02 TEST RUN START =====")
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
        cls.logger.info("===== SECOND PROD 02 TEST RUN END =====")
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
        answer_raw = response.answer_text
        answer = _normalize(answer_raw)
        answer_lines = [line.strip() for line in str(answer_raw).splitlines() if line.strip()]

        expected_topics_any = [str(item) for item in turn.get("expected_topics_any", [])]
        expected_actions_any = [str(item) for item in turn.get("expected_actions_any", [])]
        must_include_any = [str(item) for item in turn.get("must_include_any", [])]
        must_include_all = [str(item) for item in turn.get("must_include_all", [])]
        must_not_include = [str(item) for item in turn.get("must_not_include", [])]
        must_differ = bool(turn.get("must_differ_from_prev_bot", False))
        min_lines = turn.get("min_lines")
        max_lines = turn.get("max_lines")

        self.logger.info(
            "scenario=%s | turn=%s | precheck query=%s | topics=%s | action=%s | answer=%s",
            scenario_name,
            turn_index,
            user_query,
            response.topic_ids,
            response.action_name,
            answer_raw,
        )

        failures: list[str] = []

        if expected_topics_any and not any(topic in response.topic_ids for topic in expected_topics_any):
            failures.append(
                f"[{scenario_name}][turn {turn_index}] topic mismatch. expected any={expected_topics_any}, got={response.topic_ids}"
            )
        if expected_actions_any and response.action_name not in expected_actions_any:
            failures.append(
                f"[{scenario_name}][turn {turn_index}] action mismatch. expected any={expected_actions_any}, got={response.action_name}"
            )
        if must_include_any and not _contains_any(answer, must_include_any):
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_include_any failed. needles={must_include_any}, answer={answer_raw}"
            )
        if must_include_all and not _contains_all(answer, must_include_all):
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_include_all failed. needles={must_include_all}, answer={answer_raw}"
            )
        if must_not_include and _contains_any(answer, must_not_include):
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_not_include violated. needles={must_not_include}, answer={answer_raw}"
            )
        if must_differ and prev_bot_answer and answer == _normalize(prev_bot_answer):
            failures.append(
                f"[{scenario_name}][turn {turn_index}] repeated_bot_answer. answer={answer_raw}"
            )
        if isinstance(min_lines, int) and len(answer_lines) < min_lines:
            failures.append(
                f"[{scenario_name}][turn {turn_index}] too_few_lines. min_lines={min_lines}, got={len(answer_lines)}, answer={answer_raw}"
            )
        if isinstance(max_lines, int) and len(answer_lines) > max_lines:
            failures.append(
                f"[{scenario_name}][turn {turn_index}] too_many_lines. max_lines={max_lines}, got={len(answer_lines)}, answer={answer_raw}"
            )

        if response.action_name != "clarify_request":
            evidence_items = response.evidence_pack.items if response.evidence_pack else []
            if not evidence_items:
                failures.append(f"[{scenario_name}][turn {turn_index}] evidence_pack.items is empty for non-clarify action.")
            else:
                allowed_sources = _allowed_evidence_sources(response.topic_ids)
                weak_sources = [
                    str(item.source)
                    for item in evidence_items
                    if str(item.source).strip() not in allowed_sources
                ]
                if weak_sources:
                    failures.append(
                        f"[{scenario_name}][turn {turn_index}] evidence source not allowed: {weak_sources[:2]} | allowed={sorted(allowed_sources)}"
                    )
            if not response.contract_flags.get("trace_complete", False):
                failures.append(f"[{scenario_name}][turn {turn_index}] trace_complete=False")
            if not response.contract_flags.get("planned_action_matches", False):
                failures.append(f"[{scenario_name}][turn {turn_index}] planned_action_matches=False")

        if response.action_name not in self._service_actions and answer in self._template_phrases:
            failures.append(
                f"[{scenario_name}][turn {turn_index}] template leakage: answer equals policy template for non-service action."
            )

        self.logger.info(
            "scenario=%s | turn=%s | postcheck query=%s | topics=%s | failures=%s | answer=%s",
            scenario_name,
            turn_index,
            user_query,
            response.topic_ids,
            failures,
            answer_raw,
        )
        if failures:
            self.fail("\n".join(failures))
        return answer_raw


def _build_test_case(scenario: dict[str, object]):
    def test_method(self: SecondProdDialogTests02) -> None:
        session_id = f"second-prod-02-{scenario['name']}"
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
        SecondProdDialogTests02,
        f"test_second_prod_02_{_scenario['name']}",
        _build_test_case(_scenario),
    )


if __name__ == "__main__":
    unittest.main()
# cd /root/project/Chat_bot && /root/project/.venv/bin/python -m unittest tests/test_second_prod_02.py

# truncate -s 0 /root/project/Chat_bot/logs/tests_02.log
