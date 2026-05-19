from __future__ import annotations

import unittest

from the_First_Agent.parsing.direct_parser import DirectClassificationParser


class ParserFallback01Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DirectClassificationParser(
            {
                "epc_tariffs": "Тарифы EPC",
                "tis_tariffs": "Тарифы TIS",
                "specific_brand_check": "Проверка конкретного бренда",
                "human_operator_request": "Перевод на менеджера",
            }
        )

    def test_valid_intent_id_maps_to_topic_id(self) -> None:
        result = self.parser.parse(
            '{"intent_1":{"intent_id":"tis_tariffs","score":0.91,"reason":"Клиент спрашивает цену TIS."},"intent_2":null}'
        )
        self.assertEqual(result["topic_ids"], ["tis_tariffs"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "")

    def test_unknown_intent_id_sets_unknown_intent_fallback(self) -> None:
        result = self.parser.parse(
            '{"intent_1":{"intent_id":"unknown_topic","score":0.80,"reason":"Что-то выбрано."},"intent_2":null}'
        )
        self.assertEqual(result["topic_ids"], [])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "unknown_intent_id")

    def test_alias_intent_id_maps_to_canonical_topic(self) -> None:
        result = self.parser.parse(
            '{"intent_1":{"intent_id":"human_request","score":0.80,"reason":"Клиент просит человека."},"intent_2":null}'
        )
        self.assertEqual(result["topic_ids"], ["human_operator_request"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "")

    def test_invalid_json_sets_parse_fallback(self) -> None:
        result = self.parser.parse("не json вообще")
        self.assertEqual(result["topic_ids"], [])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "json_parse_failed")

    def test_old_short_json_shape_fails_schema_validation(self) -> None:
        result = self.parser.parse('{"label":"Тарифы EPC","confidence":0.95}')
        self.assertEqual(result["topic_ids"], [])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "schema_validation_failed")
        self.assertTrue(result["validation_errors"])

    def test_json_with_extra_text_is_extracted(self) -> None:
        result = self.parser.parse(
            'Вот ответ модели: {"intent_1":{"intent_id":"epc_tariffs","score":0.88,"reason":"Клиент спрашивает цену EPC."},"intent_2":null} Спасибо.'
        )
        self.assertEqual(result["topic_ids"], ["epc_tariffs"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "")

    def test_first_valid_json_is_used_when_two_blocks_exist(self) -> None:
        result = self.parser.parse(
            '{"intent_1":{"intent_id":"tis_tariffs","score":0.91,"reason":"Первый JSON."},"intent_2":null}'
            '\n'
            '{"intent_1":{"intent_id":"epc_tariffs","score":0.77,"reason":"Второй JSON."},"intent_2":null}'
        )
        self.assertEqual(result["topic_ids"], ["tis_tariffs"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "")

    def test_two_intents_keep_both_reasons(self) -> None:
        result = self.parser.parse(
            '{"intent_1":{"intent_id":"epc_tariffs","score":0.95,"reason":"Клиент спрашивает общую стоимость подписки."},'
            '"intent_2":{"intent_id":"tis_tariffs","score":0.82,"reason":"Вопрос о стоимости общий, без уточнения бренда."}}'
        )
        self.assertEqual(result["topic_ids"], ["epc_tariffs", "tis_tariffs"])
        self.assertEqual(
            result["intent_reasons"],
            [
                "Клиент спрашивает общую стоимость подписки.",
                "Вопрос о стоимости общий, без уточнения бренда.",
            ],
        )
        self.assertEqual(len(result["intent_details"]), 2)
        self.assertEqual(
            result["reason"],
            "Клиент спрашивает общую стоимость подписки.; Вопрос о стоимости общий, без уточнения бренда.",
        )

    def test_missing_reason_falls_back_to_generated_reason(self) -> None:
        result = self.parser.parse(
            '{"intent_1":{"intent_id":"tis_tariffs","score":0.89,"reason":" "},"intent_2":null}'
        )
        self.assertEqual(result["topic_ids"], ["tis_tariffs"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["reason"], "Клиентский запрос отнесен к теме `tis_tariffs`.")

    def test_missing_intent_id_fails_schema_validation(self) -> None:
        parser = DirectClassificationParser(
            {
                "brand_list_request": "Все бренды",
            }
        )
        result = parser.parse(
            '{"intent_1":{"intent_id":"","score":0.81,"reason":"Клиент просит список брендов."},"intent_2":null}'
        )
        self.assertEqual(result["topic_ids"], [])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "schema_validation_failed")


if __name__ == "__main__":
    unittest.main()
