from __future__ import annotations

import logging
import os
import re
import sys
import unittest
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.main import build_app


TIS_PRICE_BY_BRAND = {
    "audi": 6000,
    "autotech": 1000,
    "bmw": 5000,
    "daf": 10000,
    "fiat": 2000,
    "gac": 2000,
    "honda": 3000,
    "hyundai": 2000,
    "infiniti": 2000,
    "kia": 2000,
    "jaguar": 6000,
    "land rover": 6000,
    "mercedes": 10000,
    "nissan": 2000,
    "opel": 3000,
    "peugeot": 3000,
    "citroen": 3000,
    "porsche": 15000,
    "renault": 4000,
    "skoda": 6000,
    "toyota": 5000,
    "lexus": 5000,
    "volkswagen": 6000,
    "volvo": 10000,
    "zeekr": 6000,
    "lada": 2500,
    "bobcat": 6000,
}


BRAND_ALIAS_TO_CANONICAL = {
    "honda": "honda",
    "bmw": "bmw",
    "toyota": "toyota",
    "lexus": "lexus",
    "audi": "audi",
    "volkswagen": "volkswagen",
    "мерседес": "mercedes",
    "шкода": "skoda",
    "хундай": "hyundai",
    "kia": "kia",
    "renault": "renault",
    "peugeot": "peugeot",
    "nissan": "nissan",
    "infiniti": "infiniti",
    "zeekr": "zeekr",
    "zeeker": "zeekr",
    "зикер": "zeekr",
    "gac": "gac",
    "lada": "lada",
    "лада": "lada",
    "ваз": "lada",
    "ваза": "lada",
    "автоваз": "lada",
    "bobcat": "bobcat",
    "бобкат": "bobcat",
    "бобкэт": "bobcat",
}


BRAND_VARIANTS: list[tuple[str, str]] = [
    ("honda", "bmw"),
    ("toyota", "lexus"),
    ("audi", "volkswagen"),
    ("мерседес", "шкода"),
    ("хундай", "kia"),
    ("renault", "peugeot"),
    ("nissan", "infiniti"),
    ("zeekr", "gac"),
]

SEMANTIC_EQUIVALENTS = {
    "demo": [
        "demo",
        "trial",
        "демо",
        "демо доступ",
        "демо-доступ",
    ],
    "legal_only_policy": [
        "работаем только с юрид",
        "мы работаем только с юрид",
        "только юридическ",
        "ип относ",
        "юрлиц",
        "юридическ",
    ],
    "request_requisites": [
        "инн",
        "реквизит",
        "телефон",
        "имя",
        "счет",
        "счёт",
        "qr",
    ],
    "physical_not_available": [
        "не продаем доступ частным лицам",
        "доступ частным лицам мы не продаем",
        "не продаем доступ физ",
        "доступ физлицам не продаем",
        "не предоставляем демо-доступ частным",
        "частным пользователям",
        "работаем только с юридическими лицами и ип",
        "только для профессионалов",
    ],
    "price_epc": ["5500", "15000", "28800", "54000", "epc full"],
    "price_tis_context": ["tis", "техническ", "сервисн"],
    "api_not_available": [
        "нет решения для интеграции по api",
        "api",
        "интеграц",
        "доступ предоставляется только",
    ],
    "human_operator_reply": [
        "передам ваш контакт менеджеру",
        "могу ответить на все ваши вопросы",
        "могу помочь быстрее",
        "я могу оперативно предоставить вам всю информацию",
        "подключу менеджера",
    ],
    "manager_second_phone_request": [
        "укажите ваш номер телефона",
        "после этого с вами свяжется наш менеджер",
    ],
    "phone_invalid_msg": [
        "11 цифр",
        "корректный номер телефона",
    ],
    "phone_confirm_msg": [
        "номер принят",
        "менеджер свяжется",
    ],
    "parts_not_supported": [
        "подбор деталей выполняется через наши каталоги",
        "чат-бот не подбирает запчасти",
    ],
    "price_first": [
        "5500",
        "15000",
        "28800",
        "54000",
        "руб",
    ],
    "macos_reply": ["mac", "macos", "мак", "на mac os у нас софта нет", "софта нет"],
    "brand_availability": ["есть каталоги", "у нас есть каталоги", "да, у нас есть"],
    "partial_restriction": [
        "только полный доступ",
        "на отдельные бренды не продаются",
        "отдельные бренды не продаются",
        "каталоги на отдельные бренды не продаются",
    ],
    "price_objection_quality": ["качество", "актуаль", "vin", "ошиб", "убытк"],
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


def _contains_all(text: str, needles: list[str]) -> bool:
    expanded_groups = []
    for needle in needles:
        key = _normalize_text(needle)
        if key in SEMANTIC_EQUIVALENTS:
            expanded_groups.append([_normalize_text(item) for item in SEMANTIC_EQUIVALENTS[key]])
        else:
            expanded_groups.append([key])
    return all(any(variant in text for variant in group) for group in expanded_groups)


def _has_greeting_prefix(text: str) -> bool:
    return bool(re.match(r"^(здравствуйте|добрый день|добрый вечер|привет)\b", text))


def _extract_numbers(text: str) -> list[int]:
    values: list[int] = []
    for raw in re.findall(r"\d[\d\s]*", text):
        compact = re.sub(r"\s+", "", raw)
        if compact.isdigit():
            values.append(int(compact))
    return values


def _strict_total_check(response_text: str, expected_total: int) -> bool:
    normalized = _normalize_text(response_text)
    tail = normalized[-350:]
    numbers = _extract_numbers(tail)
    has_total_marker = ("итого" in tail) or ("общая сумма" in tail) or ("общ" in tail)
    return has_total_marker and (expected_total in numbers)


def _tis_total(brand_a: str, brand_b: str) -> int:
    a = TIS_PRICE_BY_BRAND[BRAND_ALIAS_TO_CANONICAL[brand_a]]
    b = TIS_PRICE_BY_BRAND[BRAND_ALIAS_TO_CANONICAL[brand_b]]
    return a + b


def _make_turn(
    user: str,
    expected_topics_any: list[str] | None = None,
    must_include_any: list[str] | None = None,
    must_include_all: list[str] | None = None,
    must_not_include: list[str] | None = None,
    strict_total: int | None = None,
) -> dict[str, object]:
    return {
        "user": user,
        "expected_topics_any": expected_topics_any or [],
        "must_include_any": must_include_any or [],
        "must_include_all": must_include_all or [],
        "must_not_include": must_not_include or [],
        "strict_total": strict_total,
    }


def _build_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    idx = 1
    for brand_a, brand_b in BRAND_VARIANTS:
        total = _tis_total(brand_a, brand_b)

        # Flow 1: legal purchase
        scenarios.append(
            {
                "name": f"{idx:02d}_legal_purchase_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Здравствуйте, нужен каталог", ["legal_status_unknown", "purchase_ready"], ["legal_only_policy"]),
                    _make_turn("Да, мы ИП", ["legal_entity_purchase_flow"], ["request_requisites"]),
                    _make_turn("Ок, выставляйте счет", ["legal_entity_purchase_flow"], ["request_requisites"]),
                    _make_turn(f"И сразу нужен {brand_a}", ["specific_brand_check", "brand_list_request"], ["brand_availability"]),
                ],
            }
        )
        idx += 1

        # Flow 2: physical person reject
        scenarios.append(
            {
                "name": f"{idx:02d}_physical_reject_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Хочу купить доступ", ["legal_status_unknown", "purchase_ready"], ["legal_only_policy"]),
                    _make_turn("Я физлицо", ["physical_person_purchase"], ["physical_not_available"]),
                    _make_turn("Почему нельзя физикам?", ["no_private_sales_reason"], ["physical_not_available"]),
                    _make_turn("Понял, спасибо", ["physical_person_purchase", "out_of_scope_request"]),
                ],
            }
        )
        idx += 1

        # Flow 3: EPC pricing and buy
        scenarios.append(
            {
                "name": f"{idx:02d}_epc_price_buy_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Сколько стоит EPC Full?", ["epc_tariffs"], ["price_epc"]),
                    _make_turn("Нужно на 3 месяца", ["epc_tariffs"], ["15000"]),
                    _make_turn("Подойдет, как купить?", ["purchase_ready", "legal_entity_purchase_flow"], ["request_requisites"]),
                    _make_turn("Можно оплатить по qr?", ["legal_entity_purchase_flow"], ["request_requisites"]),
                ],
            }
        )
        idx += 1

        # Flow 4: TIS sum strict total
        scenarios.append(
            {
                "name": f"{idx:02d}_tis_sum_strict_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn(f"Нужен TIS для {brand_a}", ["tis_tariffs"], ["price_tis_context"]),
                    _make_turn(
                        f"Добавьте {brand_b} на 1 месяц и посчитайте общую сумму",
                        ["tis_tariffs"],
                        ["итого", "общая сумма"],
                        strict_total=total,
                    ),
                    _make_turn("Отлично, и как оплатить?", ["legal_entity_purchase_flow", "purchase_ready"], ["request_requisites"]),
                ],
            }
        )
        idx += 1

        # Flow 5: API and fallback
        scenarios.append(
            {
                "name": f"{idx:02d}_api_then_fallback_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Есть интеграция по API с 1С?", ["api_integration"], ["api_not_available"]),
                    _make_turn("Ок, тогда без api, просто доступ", ["purchase_ready", "legal_entity_purchase_flow"], ["request_requisites"]),
                    _make_turn("Нужно на компанию, мы юрлицо", ["legal_entity_purchase_flow"], ["request_requisites"]),
                    _make_turn("Какие данные отправить?", ["legal_entity_purchase_flow"], ["request_requisites"]),
                ],
            }
        )
        idx += 1

        # Flow 6: demo flow
        scenarios.append(
            {
                "name": f"{idx:02d}_demo_flow_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Можно демо доступ?", ["demo_access"], ["demo"]),
                    _make_turn("Да, мы юрлицо", ["legal_entity_purchase_flow", "demo_access"], ["демо"]),
                    _make_turn("А если бы были физлицом?", ["physical_person_purchase", "no_private_sales_reason"], ["physical_not_available"]),
                    _make_turn("Тогда лучше сразу покупка", ["purchase_ready", "legal_entity_purchase_flow"], ["request_requisites"]),
                ],
            }
        )
        idx += 1

        # Flow 7: specific brand with typo
        scenarios.append(
            {
                "name": f"{idx:02d}_specific_brand_typo_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Какие бренды есть?", ["brand_list_request"], ["бренд", "каталог"]),
                    _make_turn(f"А {brand_a} и {brand_b} есть?", ["specific_brand_check"], ["brand_availability"]),
                    _make_turn("Можно только один бренд купить?", ["partial_catalog_request"], ["partial_restriction"]),
                    _make_turn("Тогда только TIS", ["tis_tariffs"], ["tis"]),
                ],
            }
        )
        idx += 1

        # Flow 8: competitor objection
        scenarios.append(
            {
                "name": f"{idx:02d}_competitor_objection_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("У конкурентов дешевле", ["price_objection"], ["price_objection_quality"]),
                    _make_turn("Тогда уйду к ним", ["competitor_choice"], ["price_objection_quality"]),
                    _make_turn("Ладно, сколько у вас на 1 месяц EPC?", ["epc_tariffs"], ["5500"]),
                    _make_turn("И TIS для bmw сколько?", ["tis_tariffs"], ["bmw", "5000"]),
                ],
            }
        )
        idx += 1

        # Flow 9: human operator
        scenarios.append(
            {
                "name": f"{idx:02d}_human_operator_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Я запутался в тарифах", ["epc_tariffs", "tis_tariffs", "nonsense_input", "company_services_info"]),
                    _make_turn("Позовите человека", ["human_operator_request"], ["human_operator_reply"]),
                    _make_turn("Хорошо, тогда подскажите про EPC", ["epc_tariffs"], ["5500"]),
                    _make_turn("И можно оплату по карте?", ["legal_entity_purchase_flow"], ["request_requisites"]),
                ],
            }
        )
        idx += 1

        # Flow 10: macOS + buy
        scenarios.append(
            {
                "name": f"{idx:02d}_macos_buy_{brand_a}_{brand_b}",
                "turns": [
                    _make_turn("Хочу купить доступ", ["purchase_ready", "legal_status_unknown"], ["legal_only_policy"]),
                    _make_turn("На маке есть приложение?", ["macos_support"], ["macos_reply"]),
                    _make_turn("Ок, оформим на ИП", ["legal_entity_purchase_flow"], ["request_requisites"]),
                    _make_turn(f"И нужен каталог {brand_a}", ["specific_brand_check", "brand_list_request"]),
                ],
            }
        )
        idx += 1

    return scenarios


def _build_extra_scenarios(start_index: int) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    idx = start_index

    scenarios.append(
        {
            "name": f"{idx:02d}_new_brand_lada_tis_price",
            "turns": [
                _make_turn("Сколько стоит TIS для Lada?", ["tis_tariffs"], ["price_tis_context"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_new_brand_vaz_alias",
            "turns": [
                _make_turn("Есть каталог ВАЗ?", ["specific_brand_check", "brand_list_request"], ["brand_availability"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_new_brand_avtovaz_alias",
            "turns": [
                _make_turn("А АвтоВАЗ есть?", ["specific_brand_check", "brand_list_request"], ["brand_availability"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_new_brand_bobcat_tis_sum",
            "turns": [
                _make_turn("Нужен TIS для Bobcat", ["tis_tariffs"], ["price_tis_context"]),
                _make_turn("Добавьте Lada и посчитайте общую сумму", ["tis_tariffs"], ["итого", "общая сумма"], strict_total=8500),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_zeekr_spelling_canonical",
            "turns": [
                _make_turn("Нужен TIS для Zeekr", ["tis_tariffs"], ["price_tis_context"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_zeeker_spelling_alias",
            "turns": [
                _make_turn("Нужен TIS для Zeeker", ["tis_tariffs"], ["price_tis_context"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_manager_first_request",
            "turns": [
                _make_turn("Позовите менеджера", ["human_operator_request"], ["human_operator_reply"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_manager_second_request_phone",
            "turns": [
                _make_turn("Нужен менеджер", ["human_operator_request"], ["human_operator_reply"]),
                _make_turn("Все равно позовите менеджера", ["human_operator_request"], ["manager_second_phone_request"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_manager_phone_invalid",
            "turns": [
                _make_turn("Позовите менеджера", ["human_operator_request"], ["human_operator_reply"]),
                _make_turn("Хочу именно менеджера", ["human_operator_request"], ["manager_second_phone_request"]),
                _make_turn("Мой номер 12345", ["human_operator_request", "nonsense_input"], ["phone_invalid_msg"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_manager_phone_valid",
            "turns": [
                _make_turn("Позовите менеджера", ["human_operator_request"], ["human_operator_reply"]),
                _make_turn("Хочу именно менеджера", ["human_operator_request"], ["manager_second_phone_request"]),
                _make_turn("Мой номер 79991234567", ["human_operator_request"], ["phone_confirm_msg"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_parts_selection_request",
            "turns": [
                _make_turn("Подбери деталь по vin", ["company_services_info"], ["parts_not_supported"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_parts_selection_article_request",
            "turns": [
                _make_turn("Подберите запчасть, артикул не знаю", ["company_services_info"], ["parts_not_supported"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_price_before_legal_status_epc",
            "turns": [
                _make_turn("Сколько стоит EPC Full, я физлицо", ["epc_tariffs", "physical_person_purchase"], ["price_first"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_price_before_legal_status_tis",
            "turns": [
                _make_turn("Я физлицо, сколько стоит TIS для Bobcat?", ["tis_tariffs", "physical_person_purchase"], ["price_tis_context"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_joke_brand_firewood",
            "turns": [
                _make_turn("Есть каталог марки дрова?", ["out_of_scope_request", "brand_list_request", "specific_brand_check"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_joke_bicycle_catalog",
            "turns": [
                _make_turn("А каталог велосипедов есть?", ["out_of_scope_request", "brand_list_request", "specific_brand_check"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_joke_spaceship_catalog",
            "turns": [
                _make_turn("Нужен каталог космических кораблей", ["out_of_scope_request"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_nonsense_random_text",
            "turns": [
                _make_turn("трали вали 123 xyz", ["nonsense_input", "out_of_scope_request"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_manager_second_then_invalid_then_valid",
            "turns": [
                _make_turn("Нужен менеджер", ["human_operator_request"], ["human_operator_reply"]),
                _make_turn("Соедините с менеджером", ["human_operator_request"], ["manager_second_phone_request"]),
                _make_turn("телефон: 999", ["human_operator_request", "nonsense_input"], ["phone_invalid_msg"]),
                _make_turn("телефон: 89001234567", ["human_operator_request"], ["phone_confirm_msg"]),
            ],
        }
    )
    idx += 1
    scenarios.append(
        {
            "name": f"{idx:02d}_parts_then_price_flow",
            "turns": [
                _make_turn("Подберите деталь на toyota", ["company_services_info"], ["parts_not_supported"]),
                _make_turn("Тогда сколько стоит EPC?", ["epc_tariffs"], ["price_epc"]),
            ],
        }
    )

    return scenarios


SCENARIOS = _build_scenarios() + _build_extra_scenarios(81)


class ChatBotE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.tests_log_path = cls.project_root / "logs/tests.log"
        cls.logger = logging.getLogger("chat_bot_e2e_tests")
        cls.logger.setLevel(logging.INFO)
        cls.logger.handlers.clear()
        cls.logger.propagate = False
        handler = logging.FileHandler(cls.tests_log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        cls.logger.addHandler(handler)
        cls.logger.info("===== CHAT BOT E2E TEST RUN START =====")

        config_path = Path(os.getenv("CHATBOT_CONFIG_PATH", str(cls.project_root / "config.yaml")))
        os.environ["CHATBOT_CONFIG_PATH"] = str(config_path)
        cls.app = build_app(cls.project_root)
        if len(SCENARIOS) != 100:
            raise AssertionError(f"Expected 100 scenarios, got {len(SCENARIOS)}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.info("===== CHAT BOT E2E TEST RUN END =====")
        for handler in cls.logger.handlers:
            handler.close()
        cls.logger.handlers.clear()

    def _run_turn(self, session_id: str, scenario_name: str, turn_index: int, turn: dict[str, object]) -> None:
        user_query = str(turn["user"])
        response = self.app.respond(session_id=session_id, user_query=user_query)
        answer_text = _normalize_text(response.answer_text)

        expected_topics_any = [str(item) for item in turn.get("expected_topics_any", [])]
        must_include_any = [str(item) for item in turn.get("must_include_any", [])]
        must_include_all = [str(item) for item in turn.get("must_include_all", [])]
        must_not_include = [str(item) for item in turn.get("must_not_include", [])]
        strict_total = turn.get("strict_total")

        checks = {
            "expected_topics_any": expected_topics_any,
            "must_include_any": must_include_any,
            "must_include_all": must_include_all,
            "must_not_include": must_not_include,
            "strict_total": strict_total,
        }
        self.logger.info(
            "scenario=%s | turn=%s | precheck query=%s | topics=%s | planned_action=%s | final_action=%s | checks=%s | answer=%s",
            scenario_name,
            turn_index,
            user_query,
            response.topic_ids,
            response.planned_action,
            response.action_name,
            checks,
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
        if must_include_all and not _contains_all(answer_text, must_include_all):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_include_all failed. "
                f"needles={must_include_all}, answer={response.answer_text}"
            )
        if must_not_include and _contains_any(answer_text, must_not_include):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] must_not_include violated. "
                f"needles={must_not_include}, answer={response.answer_text}"
            )
        if turn_index > 1 and _has_greeting_prefix(answer_text):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] repeated_greeting. "
                f"answer={response.answer_text}"
            )
        if isinstance(strict_total, int) and not _strict_total_check(answer_text, strict_total):
            response_ok = False
            failures.append(
                f"[{scenario_name}][turn {turn_index}] strict_total failed. "
                f"expected_total={strict_total}, answer={response.answer_text}"
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


def _build_test_case(scenario: dict[str, object]):
    def test_method(self: ChatBotE2ETests) -> None:
        session_id = f"e2e-{scenario['name']}"
        self.app.clear_session(session_id)
        turns = scenario["turns"]
        for idx, turn in enumerate(turns, start=1):
            self._run_turn(session_id, str(scenario["name"]), idx, turn)
        self.app.clear_session(session_id)

    return test_method


for _scenario in SCENARIOS:
    setattr(
        ChatBotE2ETests,
        f"test_chat_bot_{_scenario['name']}",
        _build_test_case(_scenario),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

# cd /root/project/Chat_bot && /root/project/.venv/bin/python -u - <<'PY'
# import subprocess
# from tests.test_chat_bot import ChatBotE2ETests

# all_tests = sorted([n for n in dir(ChatBotE2ETests) if n.startswith("test_chat_bot_")])
# BATCH_SIZE = 5      # или 10
# BASE_TIMEOUT = 180
# LONG_TIMEOUT = 360

# failed = []
# timed_out = []

# for i in range(0, len(all_tests), BATCH_SIZE):
#     batch = all_tests[i:i+BATCH_SIZE]
#     print(f"\n===== BATCH {i+1}-{i+len(batch)} =====", flush=True)

#     for name in batch:
#         test_id = f"tests.test_chat_bot.ChatBotE2ETests.{name}"
#         timeout = LONG_TIMEOUT if "tis_sum_strict_" in test_id else BASE_TIMEOUT
#         print(f"[RUN] {test_id} (timeout={timeout}s)", flush=True)
#         try:
#             r = subprocess.run(
#                 ["/root/project/.venv/bin/python", "-m", "unittest", "-v", test_id],
#                 cwd="/root/project/Chat_bot",
#                 timeout=timeout
#             )
#             if r.returncode != 0:
#                 failed.append(test_id)
#                 print(f"[FAIL] {test_id}", flush=True)
#             else:
#                 print(f"[OK]   {test_id}", flush=True)
#         except subprocess.TimeoutExpired:
#             timed_out.append(test_id)
#             print(f"[TIMEOUT] {test_id}", flush=True)

# print("\n===== SUMMARY =====")
# print(f"TOTAL: {len(all_tests)}")
# print(f"FAILED: {len(failed)}")
# print(f"TIMEOUT: {len(timed_out)}")

# if failed:
#     print("\nFailed tests:")
#     for t in failed:
#         print(" -", t)

# if timed_out:
#     print("\nTimed out tests:")
#     for t in timed_out:
#         print(" -", t)
# PY

# ========================================================================================================

# cd /root/project/Chat_bot && CHATBOT_API_KEY="xK9mLpQ2vN7wR" /root/project/.venv/bin/python -u - <<'PY'
# import subprocess
# from tests.test_chat_bot import ChatBotE2ETests


# all_tests = sorted([n for n in dir(ChatBotE2ETests) if n.startswith("test_chat_bot_")])
# BATCH_SIZE = 5      # или 10
# BASE_TIMEOUT = 180
# LONG_TIMEOUT = 360


# failed = []
# timed_out = []


# for i in range(0, len(all_tests), BATCH_SIZE):
#     batch = all_tests[i:i+BATCH_SIZE]
#     print(f"\n===== BATCH {i+1}-{i+len(batch)} =====", flush=True)


#     for name in batch:
#         test_id = f"tests.test_chat_bot.ChatBotE2ETests.{name}"
#         timeout = LONG_TIMEOUT if "tis_sum_strict_" in test_id else BASE_TIMEOUT
#         print(f"[RUN] {test_id} (timeout={timeout}s)", flush=True)
#         try:
#             r = subprocess.run(
#                 ["/root/project/.venv/bin/python", "-m", "unittest", "-v", test_id],
#                 cwd="/root/project/Chat_bot",
#                 timeout=timeout
#             )
#             if r.returncode != 0:
#                 failed.append(test_id)
#                 print(f"[FAIL] {test_id}", flush=True)
#             else:
#                 print(f"[OK]   {test_id}", flush=True)
#         except subprocess.TimeoutExpired:
#             timed_out.append(test_id)
#             print(f"[TIMEOUT] {test_id}", flush=True)


# print("\n===== SUMMARY =====")
# print(f"TOTAL: {len(all_tests)}")
# print(f"FAILED: {len(failed)}")
# print(f"TIMEOUT: {len(timed_out)}")


# if failed:
#     print("\nFailed tests:")
#     for t in failed:
#         print(" -", t)


# if timed_out:
#     print("\nTimed out tests:")
#     for t in timed_out:
#         print(" -", t)
# PY
