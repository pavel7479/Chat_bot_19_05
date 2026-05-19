from __future__ import annotations

import unittest

from the_First_Agent.parsing.direct_parser import DirectClassificationParser


class ParserLabelNormalization01Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DirectClassificationParser(
            {
                "tis_tariffs": "Тарифы TIS",
                "brand_list_request": "Все бренды",
            }
        )

    def test_intent_id_is_source_of_truth_without_label_ru_field(self) -> None:
        result_1 = self.parser.parse(
            '{"intent_1":{"intent_id":"tis_tariffs","score":0.90,"reason":"Цена TIS."},"intent_2":null}'
        )
        result_2 = self.parser.parse(
            '{"intent_1":{"intent_id":"brand_list_request","score":0.80,"reason":"Список брендов."},"intent_2":null}'
        )

        self.assertEqual(result_1["topic_ids"], ["tis_tariffs"])
        self.assertFalse(result_1["fallback_used"])

        self.assertEqual(result_2["topic_ids"], ["brand_list_request"])
        self.assertFalse(result_2["fallback_used"])


if __name__ == "__main__":
    unittest.main()
