import unittest

from src.agents.response_policy import ResponseActionSelector, ResponseState


class ResponseActionSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = ResponseActionSelector()
        self.state = ResponseState()

    def test_price_query_prefers_tis_when_tis_is_explicit(self) -> None:
        action = self.selector.select(
            topic_ids=["tis_tariffs", "epc_tariffs"],
            state=self.state,
            user_query="тис лада сколько стоит",
            history_text="user: сколько стоит epc",
        )
        self.assertEqual(action.name, "tis_tariffs")

    def test_price_query_prefers_epc_when_epc_is_explicit(self) -> None:
        action = self.selector.select(
            topic_ids=["tis_tariffs", "epc_tariffs"],
            state=self.state,
            user_query="а epc сколько стоит",
            history_text="user: подскажите tis",
        )
        self.assertEqual(action.name, "epc_tariffs")

    def test_price_query_uses_primary_when_both_topics_present_without_markers(self) -> None:
        action = self.selector.select(
            topic_ids=["tis_tariffs", "epc_tariffs"],
            state=self.state,
            user_query="сколько стоит",
            history_text="",
        )
        self.assertEqual(action.name, "tis_tariffs")

    def test_compare_query_returns_compare_action(self) -> None:
        action = self.selector.select(
            topic_ids=["epc_tariffs", "tis_tariffs"],
            state=self.state,
            user_query="в чем разница между epc и tis",
            history_text="",
        )
        self.assertEqual(action.name, "compare_epc_tis")

    def test_phone_confirm_after_invalid_prompt_without_manager_keyword(self) -> None:
        action = self.selector.select(
            topic_ids=["human_operator_request"],
            state=self.state,
            user_query="телефон: 89001234567",
            history_text=(
                "user: телефон: 999\n"
                "assistant: Пожалуйста, укажите корректный номер телефона: должно быть 11 цифр.\n"
                "user: телефон: 89001234567"
            ),
        )
        self.assertEqual(action.name, "human_operator_phone_confirm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
