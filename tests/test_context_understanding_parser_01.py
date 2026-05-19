from __future__ import annotations

import unittest

from the_First_Agent.Agent_Zero.parser import ContextUnderstandingParser


class ContextUnderstandingParser01Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ContextUnderstandingParser()
        self.user_query = "да"


def _make_parser_success_test(raw_response: str):
    def _test(self: ContextUnderstandingParser01Tests) -> None:
        result = self.parser.parse(raw_response=raw_response, user_query=self.user_query)
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.gist)
        self.assertTrue(result.meaning)

    return _test


def _make_parser_fallback_test(raw_response: str):
    def _test(self: ContextUnderstandingParser01Tests) -> None:
        result = self.parser.parse(raw_response=raw_response, user_query=self.user_query)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.meaning, f"Последняя реплика клиента: {self.user_query}")

    return _test


_SUCCESS_CASES = [
    ('{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."}', "plain_json"),
    ('  {"gist":"Диалог о цене.","meaning":"Клиент спрашивает стоимость."}  ', "json_with_whitespace"),
    ('{"gist":"   Диалог о цене.   ","meaning":"   Клиент спрашивает стоимость.   "}', "json_with_stripped_fields"),
    ('prefix {"gist":"Диалог о бренде.","meaning":"Клиент уточняет бренд."}', "text_before_json"),
    ('{"gist":"Диалог о покупке.","meaning":"Клиент готов купить."} suffix', "text_after_json"),
    ('prefix {"gist":"Диалог о TIS.","meaning":"Клиент переключился на TIS."} suffix', "text_around_json"),
    ('```json\n{"gist":"Диалог о статусе.","meaning":"Клиент подтверждает статус."}\n```', "code_fence"),
    ('{"gist":"Короткий диалог.","meaning":"Ответ является подтверждением."}\n\nлишний хвост', "tail_after_json"),
    ('noise\nnoise\n{"gist":"Диалог о Volvo.","meaning":"Клиент спрашивает цену для Volvo."}', "multiline_prefix"),
]

_FALLBACK_CASES = [
    ("", "empty_response", "empty_response"),
    ("not json", "plain_text", "json_object_not_found"),
    ('{"meaning":"Клиент подтвердил статус."}', "missing_gist", "schema_validation_failed"),
    ('{"gist":"Диалог о статусе."}', "missing_meaning", "schema_validation_failed"),
    ('{"gist":"","meaning":"Клиент подтвердил статус."}', "empty_gist", "empty_required_field"),
    ('{"gist":"Диалог о статусе.","meaning":""}', "empty_meaning", "empty_required_field"),
    ('{"gist":"   ","meaning":"Клиент подтвердил статус."}', "blank_gist", "empty_required_field"),
    ('{"gist":"Диалог о статусе.","meaning":"   "}', "blank_meaning", "empty_required_field"),
    ('{"gist":123,"meaning":"Клиент подтвердил статус."}', "gist_wrong_type", "schema_validation_failed"),
    ('{"gist":"Диалог о статусе.","meaning":123}', "meaning_wrong_type", "schema_validation_failed"),
    ('{"gist":"Диалог о статусе.","meaning":"Клиент подтвердил статус.","extra":"x"}', "extra_field", "schema_validation_failed"),
    ('[]', "json_array", "json_root_is_not_object"),
    ('{"gist":"Диалог","meaning":"Смысл"', "broken_json", "json_object_not_found"),
    ('prefix [1,2,3] suffix', "non_object_json", "json_root_is_not_object"),
]


class ContextUnderstandingParserDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ContextUnderstandingParser()

    def test_wrapped_json_sets_extraction_flag(self) -> None:
        result = self.parser.parse(
            raw_response='prefix {"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."} suffix',
            user_query="да",
        )
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.json_extracted_from_wrapped_response)

    def test_plain_json_does_not_set_extraction_flag(self) -> None:
        result = self.parser.parse(
            raw_response='{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."}',
            user_query="да",
        )
        self.assertFalse(result.fallback_used)
        self.assertFalse(result.json_extracted_from_wrapped_response)

    def test_extra_field_keeps_validation_error(self) -> None:
        result = self.parser.parse(
            raw_response='{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус.","extra":"x"}',
            user_query="да",
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "schema_validation_failed")
        self.assertIn("schema_validation_failed", result.validation_error)

    def test_empty_response_has_specific_reason(self) -> None:
        result = self.parser.parse(raw_response="", user_query="да")
        self.assertEqual(result.fallback_reason, "empty_response")

    def test_empty_fields_have_specific_reason(self) -> None:
        result = self.parser.parse(
            raw_response='{"gist":"","meaning":"Клиент подтвердил статус."}',
            user_query="да",
        )
        self.assertEqual(result.fallback_reason, "empty_required_field")


for index, (raw_response, label) in enumerate(_SUCCESS_CASES, start=1):
    setattr(
        ContextUnderstandingParser01Tests,
        f"test_success_{index:02d}_{label}",
        _make_parser_success_test(raw_response),
    )

for index, (raw_response, label, expected_reason) in enumerate(_FALLBACK_CASES, start=1):
    def _make_reasoned_test(raw_response: str, expected_reason: str):
        def _test(self: ContextUnderstandingParser01Tests) -> None:
            result = self.parser.parse(raw_response=raw_response, user_query=self.user_query)
            self.assertTrue(result.fallback_used)
            self.assertEqual(result.fallback_reason, expected_reason)
            self.assertEqual(result.meaning, f"Последняя реплика клиента: {self.user_query}")

        return _test

    setattr(
        ContextUnderstandingParser01Tests,
        f"test_fallback_{index:02d}_{label}",
        _make_reasoned_test(raw_response, expected_reason),
    )


if __name__ == "__main__":
    unittest.main()
