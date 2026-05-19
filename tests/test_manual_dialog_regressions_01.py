from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.main import build_app


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()


def _contains_any(text: str, needles: list[str]) -> bool:
    hay = _normalize(text)
    return any(_normalize(needle) in hay for needle in needles)


def _stringify(value: object) -> str:
    if isinstance(value, list):
        return " ".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key}:{_stringify(item)}" for key, item in value.items())
    return str(value)


class ManualDialogRegression01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.app = build_app(cls.project_root)

    def tearDown(self) -> None:
        self.app.clear_session(self._testMethodName)

    def _respond(self, message: str):
        response = self.app.respond(session_id=self._testMethodName, user_query=message)
        debug = self.app.get_debug_trace(self._testMethodName)
        state = self.app._session.get_state(self._testMethodName)
        return response, debug, state

    def _assert_common(self, response, expected_topics: list[str], expected_action: str | None = None) -> None:
        if expected_topics:
            self.assertTrue(
                any(topic in response.topic_ids for topic in expected_topics),
                f"Expected any topic from {expected_topics}, got {response.topic_ids}",
            )
        if expected_action is not None:
            self.assertEqual(
                response.action_name,
                expected_action,
                f"Expected action={expected_action}, got action={response.action_name}, answer={response.answer_text}",
            )

    def _assert_must_include(self, response_text: str, values: list[str]) -> None:
        if not values:
            return
        for value in values:
            self.assertTrue(
                _contains_any(response_text, [value]),
                f"Expected answer to include `{value}`, got answer={response_text}",
            )

    def _assert_must_not_include(self, response_text: str, values: list[str]) -> None:
        if not values:
            return
        for value in values:
            self.assertFalse(
                _contains_any(response_text, [value]),
                f"Expected answer to avoid `{value}`, got answer={response_text}",
            )

    def test_manual_01_greeting_once(self) -> None:
        first, _, state = self._respond("привет")
        self._assert_common(first, ["nonsense_input"], "greeting_once")
        self.assertTrue(_contains_any(first.answer_text, ["здравствуйте", "добрый", "чем могу помочь"]))
        self.assertTrue(state.greeted)

        second, _, _ = self._respond("сколько стоит подписка")
        self.assertFalse(_normalize(second.answer_text).startswith("здравствуйте"))

    def test_manual_02_catalog_list(self) -> None:
        response, debug, _ = self._respond("какие каталоги у вас есть")
        self._assert_common(response, ["catalog_list_request"], "catalog_list_info")
        self._assert_must_include(response.answer_text, ["EPC", "TIS"])
        self._assert_must_not_include(response.answer_text, ["и т.д.", "уточните марку"])
        fact_ids = debug.get("response_plan", {}).get("required_fact_ids", [])
        self.assertIn("catalog_list_products", fact_ids)

    def test_manual_03_general_pricing(self) -> None:
        response, debug, _ = self._respond("хочу доступ к каталогам что по деньгам")
        self.assertTrue(any(topic in response.topic_ids for topic in ["purchase_ready", "epc_tariffs", "tis_tariffs"]))
        self.assertEqual(debug.get("response_plan", {}).get("primary_action"), "pricing_summary")
        self.assertEqual(set(debug.get("response_plan", {}).get("required_price_blocks", [])), {"epc", "tis"})
        self._assert_must_include(response.answer_text, ["EPC", "TIS"])

    def test_manual_04_buy_subscription_process(self) -> None:
        response, _, _ = self._respond("как купить подписку")
        self._assert_common(response, ["purchase_ready"], "ask_legal_status")
        self._assert_must_include(response.answer_text, ["юридическими лицами", "ип"])
        self._assert_must_not_include(response.answer_text, ["6500", "18000"])

    def test_manual_05_payment_process(self) -> None:
        response, _, _ = self._respond("как оплатить")
        self._assert_common(response, ["payment_process"], "payment_process")
        self._assert_must_include(response.answer_text, ["счет", "qr"])
        self._assert_must_not_include(response.answer_text, ["epc full доступен на"])

    def test_manual_06_renewal(self) -> None:
        response, _, _ = self._respond("как продлить подписку")
        self._assert_common(response, ["subscription_renewal"], "subscription_renewal")
        self._assert_must_include(response.answer_text, ["продлен"])
        self._assert_must_not_include(response.answer_text, ["6500"])

    def test_manual_07_self_employed(self) -> None:
        response, _, _ = self._respond("вы работаете с самозанятыми")
        self._assert_common(response, ["self_employed_purchase"], "self_employed_policy")
        self._assert_must_include(response.answer_text, ["юридическими лицами", "ип"])
        self._assert_must_not_include(response.answer_text, ["зависит от ваших задач"])

    def test_manual_08_existing_contract(self) -> None:
        response, _, _ = self._respond("мне нужно проверить существующий договор")
        self._assert_common(response, ["existing_contract_check"], "existing_contract_check")
        self._assert_must_include(response.answer_text, ["номер договора"])
        self._assert_must_not_include(response.answer_text, ["инн", "количество доступов"])

    def test_manual_09_phone_as_requisites_followup(self) -> None:
        first, _, state = self._respond("я ИП как начать работать с вами")
        self.assertEqual(first.action_name, "request_requisites")
        self.assertEqual(state.active_request_kind, "checkout")

        second, debug, _ = self._respond("+79211234567")
        slot_trace = debug.get("topic_result_diagnostics", {}).get("slot_extraction_trace", {})
        self.assertEqual(slot_trace.get("slots", {}).get("phone"), "+79211234567")
        self.assertIn("legal_entity_purchase_flow", second.topic_ids)
        self._assert_must_not_include(second.answer_text, ["что именно вас интересует"])

    def test_manual_10_full_requisites_in_one_message(self) -> None:
        text = "Здравствуйте интересует доступ к каталогам Скания на 1 год, для 1 пользователя, ИНН 1234567890, тел. +79117456123, оплата по счёту"
        response, debug, _ = self._respond(text)
        slot_trace = debug.get("topic_result_diagnostics", {}).get("slot_extraction_trace", {})
        slots = slot_trace.get("slots", {})
        self.assertEqual(slots.get("brand"), "scania")
        self.assertEqual(slots.get("period"), "12_months")
        self.assertEqual(slots.get("user_count"), 1)
        self.assertEqual(slots.get("inn"), "1234567890")
        self.assertEqual(slots.get("phone"), "+79117456123")
        self.assertEqual(slots.get("payment_method"), "invoice")
        self._assert_must_not_include(response.answer_text, ["пришлите инн", "пришлите телефон"])

    def test_manual_11_brand_availability_then_price_followup_haval(self) -> None:
        first, _, state = self._respond("Есть ли у вас каталог на Haval Dargo")
        self._assert_common(first, ["specific_brand_check"])
        self.assertEqual(state.last_mentioned_brand, "haval")
        self._assert_must_include(first.answer_text, ["haval"])

        second, debug, _ = self._respond("подскажи")
        followup = debug.get("topic_result_diagnostics", {}).get("followup_trace", {})
        self.assertEqual(followup.get("followup_type"), "brand_price_followup")
        self.assertEqual(followup.get("inherited_brand"), "haval")
        price_context = debug.get("answer_block", {}).get("price_context", {})
        self.assertEqual(price_context.get("tis_price_status"), "missing")
        self.assertEqual(price_context.get("missing_tis_price_brands"), ["haval"])
        self.assertIn("epc", price_context.get("fallback_price_blocks", []))
        self._assert_must_include(second.answer_text, ["цена", "прайсе", "epc full"])
        self._assert_must_not_include(second.answer_text, ["уточните бренд", "уточните марку"])

    def test_manual_12_brand_availability_then_price_followup_mercedes(self) -> None:
        first, _, state = self._respond("Какой каталог мне нужен для Mercedes?")
        self.assertIn(state.last_mentioned_brand, {"mercedes-benz", "mercedes"})

        second, debug, _ = self._respond("подскажи")
        answer_prompt = _stringify(debug.get("answer_prompt", ""))
        self.assertTrue(_contains_any(answer_prompt, ["mercedes"]))
        self._assert_must_not_include(second.answer_text, ["уточните"])

    def test_manual_13_specific_brand_uaz(self) -> None:
        response, _, state = self._respond("есть ли каталог уаз")
        self._assert_common(response, ["specific_brand_check"])
        self.assertEqual(state.last_mentioned_brand, "uaz")
        self._assert_must_include(response.answer_text, ["uaz"])
        self._assert_must_not_include(response.answer_text, ["уаз доступен", "уаз доступна", "уаз доступно", "уаз "])
        self._assert_must_not_include(response.answer_text, ["уточните марку", "напишите бренд"])

    def test_manual_14_only_uaz_followup(self) -> None:
        self._respond("есть ли каталог уаз")
        response, debug, _ = self._respond("только уаз")
        followup = debug.get("topic_result_diagnostics", {}).get("followup_trace", {})
        self.assertTrue(any(topic in response.topic_ids for topic in ["partial_catalog_request", "tis_tariffs", "specific_brand_check"]))
        self.assertIn(followup.get("inherited_brand"), {"uaz", ""})
        self._assert_must_not_include(response.answer_text, ["уточните марку", "напишите бренд"])

    def test_manual_15_partial_catalog_request(self) -> None:
        response, _, _ = self._respond("есть ли возможность предоставить каталоги только одного бренда")
        self._assert_common(response, ["partial_catalog_request"], "partial_catalog_restriction")
        self._assert_must_include(response.answer_text, ["EPC Full", "полным пакетом"])

    def test_manual_16_vag_group(self) -> None:
        first, debug, _ = self._respond("только vag")
        self.assertTrue(any(topic in first.topic_ids for topic in ["partial_catalog_request", "specific_brand_check", "tis_tariffs"]))
        self.assertEqual(first.action_name, "brand_group_clarification")
        self._assert_must_include(first.answer_text, ["vag", "уточните", "конкретную марку"])
        self._assert_must_not_include(first.answer_text, ["тариф", "руб", "epc full"])

    def test_manual_17_multi_user(self) -> None:
        response, _, _ = self._respond("можно ли пользоваться нескольким людям")
        self._assert_common(response, ["multi_device_access"], "multi_device_access_info")
        self._assert_must_include(response.answer_text, ["нескольк", "доступ"])
        self._assert_must_not_include(response.answer_text, ["epc full доступен на"])

    def test_manual_18_competitor_comparison(self) -> None:
        response, _, _ = self._respond("а в чём отличие от конкурентов")
        self._assert_common(response, ["competitor_comparison"], "competitor_comparison")
        self.assertTrue(_contains_any(response.answer_text, ["актуаль", "полнот", "поддерж", "стабил"]))
        self._assert_must_not_include(response.answer_text, ["epc отличается от tis"])

    def test_manual_19_service_advantages(self) -> None:
        response, _, _ = self._respond("какие преимущества вашей подписки")
        self._assert_common(response, ["company_services_info"], "company_services")
        self.assertTrue(_contains_any(response.answer_text, ["актуаль", "поддерж", "автобизнес"]))
        self._assert_must_not_include(response.answer_text, ["уточните марку"])

    def test_manual_20_service_cons(self) -> None:
        response, debug, _ = self._respond("какие минусы у сервиса")
        self._assert_common(response, ["company_services_info"], "company_services")
        fact_ids = debug.get("response_plan", {}).get("required_fact_ids", [])
        self.assertIn("company_services_cons", fact_ids)
        self.assertTrue(_contains_any(response.answer_text, ["автобизнес", "оформлен", "ограничен"]))
        self._assert_must_not_include(response.answer_text, ["широкий список брендов"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
# cd /root/project/Chat_bot && /root/project/.venv/bin/python -m unittest -v tests.test_manual_dialog_regressions_01
# truncate -s 0 /root/project/Chat_bot/logs/chatbot.log
