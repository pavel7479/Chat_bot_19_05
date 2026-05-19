from __future__ import annotations

import unittest

from src.core.models import SessionState
from the_First_Agent.context.prompt_context import PromptContext
from the_First_Agent.preprocessing.query_rewriter import QueryRewriteService


class QueryRewriter01Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rewriter = QueryRewriteService(["volvo", "bmw"])

    @staticmethod
    def _context(user_query: str, last_assistant: str = "", history_text: str = "") -> PromptContext:
        return PromptContext(
            history_text=history_text,
            user_query=user_query,
            history_lines=[],
            last_assistant_message=last_assistant,
        )

    def test_regular_query_is_kept_as_is(self) -> None:
        context = self._context("Оплачу, но ИНН не дам", "Напишите ИНН и реквизиты для счета")
        result = self.rewriter.rewrite(context, SessionState(), "Оплачу, но ИНН не дам")
        self.assertEqual(result["mode"], "direct")
        self.assertEqual(result["rewritten_query"], "Оплачу, но ИНН не дам")

    def test_short_answer_is_not_rewritten_anymore(self) -> None:
        context = self._context("yes", "Вы юридическое лицо?")
        result = self.rewriter.rewrite(context, SessionState(), "yes")
        self.assertEqual(result["mode"], "direct")
        self.assertEqual(result["rewritten_query"], "yes")

    def test_buy_as_physical_person_kept_as_is(self) -> None:
        context = self._context("Хочу купить как физ лицо", "")
        result = self.rewriter.rewrite(context, SessionState(), "Хочу купить как физ лицо")
        self.assertEqual(result["mode"], "direct")
        self.assertEqual(result["rewritten_query"], "Хочу купить как физ лицо")

    def test_stale_session_brand_not_used_for_generic_query(self) -> None:
        context = self._context("Сколько стоит подписка?", "", "user: есть Volvo?\nassistant: да, есть")
        result = self.rewriter.rewrite(context, SessionState(last_mentioned_brand="Volvo"), "Сколько стоит подписка?")
        self.assertEqual(result["brand_resolution_trace"], {"source": "none", "brand": ""})

    def test_anaphora_uses_session_brand(self) -> None:
        context = self._context("Сколько стоит на него?", "", "user: есть Volvo?\nassistant: да, есть")
        result = self.rewriter.rewrite(context, SessionState(last_mentioned_brand="Volvo"), "Сколько стоит на него?")
        self.assertEqual(result["mode"], "anaphora_followup")
        self.assertEqual(result["brand_resolution_trace"], {"source": "anaphora", "brand": "Volvo"})
        self.assertIn("Volvo", result["rewritten_query"])


if __name__ == "__main__":
    unittest.main()
