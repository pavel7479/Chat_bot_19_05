from __future__ import annotations

import unittest
from pathlib import Path

from src.config.loader import ConfigLoader
from the_First_Agent.Agent_Zero.context_understanding_agent import ContextUnderstandingAgent


class _DummyLLM:
    def generate_text(self, prompt: str) -> str:
        return ""

    def generate_json(self, prompt: str) -> str:
        return '{"gist":"x","meaning":"y"}'


class ContextUnderstandingPrompt01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.prompt_path = cls.project_root / cls.config.paths.context_understanding_prompt
        cls.agent = ContextUnderstandingAgent(_DummyLLM(), cls.prompt_path)
        cls.prompt = cls.agent.build_prompt(
            dialog_text="бот: Вы юрлицо?\nклиент: да",
            user_query="да",
        )


def _assert_prompt_contains(fragment: str):
    def _test(self: ContextUnderstandingPrompt01Tests) -> None:
        self.assertIn(fragment, self.prompt)

    return _test


_PROMPT_ASSERTIONS = [
    ("Диалог:", "dialog_section"),
    ("Последняя реплика клиента:", "user_query_section"),
    ("Формат ответа:", "format_section"),
    ('"gist": "краткий контекст диалога"', "gist_field"),
    ('"meaning": "смысл последней реплики клиента"', "meaning_field"),
    ("Не используй intent_id, внутренние названия тем и технические термины.", "no_intent_ids_rule"),
    ("`meaning` важнее `gist`.", "meaning_priority_rule"),
    ("Не тащи старые темы, если они больше не нужны для понимания последней реплики.", "gist_history_limit_rule"),
    ("Не используй названия системных тем вроде", "no_system_topic_labels_rule"),
    ("`gist` должен быть одним коротким предложением.", "gist_length_rule"),
    ("`meaning` должен быть одним коротким предложением.", "meaning_length_rule"),
    ("Не добавляй текст вне JSON.", "json_only_rule"),
    ("Ответ должен начинаться с `{`", "starts_with_brace_rule"),
    ("Ответ должен заканчиваться `}`", "ends_with_brace_rule"),
    ("Короткие и контекстные реплики:", "short_reply_section"),
    ('"semantic_frame": {', "semantic_frame_section"),
    ('"conversation_mode": "unknown|discovery|product_choice|pricing|purchase|support|renewal|manager|complaint|security|out_of_scope|smalltalk"', "semantic_frame_conversation_mode"),
    ('"user_goal": "greet|ask_product_list|ask_product_recommendation|ask_price|ask_purchase_steps|ask_support|ask_manager|ask_benefits|ask_limitations|complain_or_distrust|ask_free_or_bypass_payment|unknown"', "semantic_frame_user_goal"),
    ('"is_followup": false', "semantic_frame_is_followup"),
    ('"is_topic_switch": false', "semantic_frame_is_topic_switch"),
    ('"language": "ru|en"', "semantic_frame_language"),
]


for index, (fragment, label) in enumerate(_PROMPT_ASSERTIONS, start=1):
    setattr(
        ContextUnderstandingPrompt01Tests,
        f"test_prompt_{index:02d}_{label}",
        _assert_prompt_contains(fragment),
    )


if __name__ == "__main__":
    unittest.main()
