from __future__ import annotations

import unittest
from pathlib import Path

from src.config.loader import ConfigLoader
from src.core.interfaces import LLMProvider
from the_First_Agent.Agent_Zero.context_understanding_agent import ContextUnderstandingAgent


class _ScriptedContextLLM(LLMProvider):
    def __init__(self, raise_on_call: int | None = None) -> None:
        self.prompts: list[str] = []
        self.raise_on_call = raise_on_call
        self.call_count = 0
        self.last_response = ""

    def generate_text(self, prompt: str) -> str:
        return ""

    def generate_json(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        if self.raise_on_call is not None and self.call_count == self.raise_on_call:
            raise RuntimeError("fake llm error")
        user_query = self._extract_user_query(prompt)
        cases = [
            ("yes, legal", '{"gist":"Клиент интересуется демо-доступом, бот уточняет статус клиента.","meaning":"Клиент подтвердил, что является юридическим лицом для получения демо-доступа."}'),
            ("нет", '{"gist":"Клиент интересуется демо-доступом, бот уточняет статус клиента.","meaning":"Клиент отрицательно ответил на вопрос о юридическом статусе в контексте демо."}'),
            ("....548", '{"gist":"Бот уточняет статус клиента для демо-доступа.","meaning":"Последняя реплика выглядит как шумовой или нерелевантный ответ и не отвечает на вопрос о статусе."}'),
            ("Либо тогда только TIS", '{"gist":"Обсуждался частичный доступ к EPC, затем клиент сменил тему.","meaning":"Клиент переключился на отдельный продукт TIS."}'),
            ("а сколько на него TIS?", '{"gist":"Клиент уточняет цену TIS для ранее упомянутого бренда Volvo.","meaning":"Клиент спрашивает стоимость TIS для Volvo."}'),
            ("Да, являюсь", '{"gist":"Бот уточняет статус клиента перед оформлением.","meaning":"Клиент подтвердил статус юридического лица или представителя автобизнеса."}'),
            ("no", '{"gist":"Бот уточняет статус клиента перед демо.","meaning":"Клиент отрицает, что относится к требуемому статусу."}'),
            ("legal", '{"gist":"Бот уточняет статус клиента.","meaning":"Клиент кратко подтверждает юридический статус."}'),
            ("все", '{"gist":"Бот уточняет бренды для расчета TIS.","meaning":"Клиент просит показать все доступные бренды."}'),
            ("а какие есть", '{"gist":"Бот уточняет бренды для расчета TIS.","meaning":"Клиент просит перечислить доступные бренды."}'),
            ("на него", '{"gist":"Клиент обсуждает ранее названный бренд.","meaning":"Клиент имеет в виду ранее упомянутый бренд и продолжает запрос по нему."}'),
            ("для него", '{"gist":"Клиент обсуждает ранее названный бренд.","meaning":"Клиент продолжает запрос применительно к ранее упомянутому бренду."}'),
            ("TIS", '{"gist":"Клиент спрашивает про продукт TIS.","meaning":"Клиент делает короткий самостоятельный продуктовый запрос про TIS."}'),
            ("EPC", '{"gist":"Клиент спрашивает про продукт EPC.","meaning":"Клиент делает короткий самостоятельный продуктовый запрос про EPC."}'),
            ("демо", '{"gist":"Клиент интересуется тестовым доступом.","meaning":"Клиент коротко запрашивает демо-доступ."}'),
            ("Почему не продаете физлицам?", '{"gist":"Клиент обсуждает условия продажи.","meaning":"Клиент спрашивает причину отказа в продаже физическим лицам."}'),
            ("Хочу купить как физ лицо", '{"gist":"Клиент обсуждает покупку доступа.","meaning":"Клиент хочет купить доступ как физическое лицо."}'),
            ("вы мошенники", '{"gist":"Клиент выражает резкое недоверие к компании.","meaning":"Клиент делает конфликтное обвинительное заявление, не задавая предметный вопрос."}'),
            ("мне обещали бесплатно навсегда", '{"gist":"Клиент обсуждает условия бесплатного доступа.","meaning":"Клиент утверждает, что ему обещали бесплатный доступ без ограничения по времени."}'),
            ("PERSISTENT_INVALID_JSON_CASE", 'невалидный ответ'),
            ("RETRY_EXTRA_FIELD_CASE", '{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус.","extra":"x"}' if self.call_count == 1 else '{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."}'),
            ("RETRY_EMPTY_RESPONSE_CASE", '' if self.call_count == 1 else '{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."}'),
            ("RETRY_JSON_OBJECT_NOT_FOUND_CASE", 'невалидный ответ' if self.call_count == 1 else '{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."}'),
            ("RETRY_JSON_DECODE_CASE", '{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус"' if self.call_count == 1 else '{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."}'),
            ("RETRY_EMPTY_REQUIRED_FIELD_CASE", '{"gist":"   ","meaning":"Клиент подтвердил статус."}' if self.call_count == 1 else '{"gist":"Диалог о демо.","meaning":"Клиент подтвердил статус."}'),
        ]
        for needle, response in cases:
            if needle == user_query:
                self.last_response = response
                return response
        self.last_response = '{"gist":"Общий диалог.","meaning":"Клиент продолжает разговор по текущей теме."}'
        return self.last_response

    @staticmethod
    def _extract_user_query(prompt: str) -> str:
        marker = "Последняя реплика клиента:\n"
        tail = prompt.split(marker, 1)[1] if marker in prompt else prompt
        return tail.strip().splitlines()[0].strip()


class ContextUnderstandingAgent01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.prompt_path = cls.project_root / cls.config.paths.context_understanding_prompt

    def setUp(self) -> None:
        self.llm = _ScriptedContextLLM()
        self.agent = ContextUnderstandingAgent(self.llm, self.prompt_path)


def _make_agent_keyword_test(dialog_text: str, user_query: str, expected_keywords: tuple[str, ...]):
    def _test(self: ContextUnderstandingAgent01Tests) -> None:
        result = self.agent.understand(dialog_text=dialog_text, user_query=user_query)
        self.assertFalse(result.fallback_used)
        lowered_meaning = result.meaning.lower()
        for keyword in expected_keywords:
            self.assertIn(keyword, lowered_meaning)
        self.assertIn(dialog_text, self.llm.prompts[-1])
        self.assertIn(user_query, self.llm.prompts[-1])

    return _test


def _make_agent_fallback_test(user_query: str):
    def _test(self: ContextUnderstandingAgent01Tests) -> None:
        result = self.agent.understand(
            dialog_text="бот: Мы можем предоставить демо только юрлицам.",
            user_query=user_query,
        )
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.fallback_reason)
        self.assertEqual(result.meaning, f"Последняя реплика клиента: {user_query}")

    return _test


_INTEGRATION_CASES = [
    ("бот: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?", "yes, legal", ("подтверд", "юрид", "демо")),
    ("бот: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?", "нет", ("отриц", "статус", "демо")),
    ("бот: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?", "....548", ("шум", "нерелевант", "вопрос")),
    ("клиент: Можно только один бренд?\nбот: EPC Full продается только полным пакетом.", "Либо тогда только TIS", ("переключ", "tis")),
    ("клиент: есть Volvo?\nбот: есть", "а сколько на него TIS?", ("стоим", "tis", "volvo")),
    ("бот: Вы юрлицо?", "Да, являюсь", ("подтверд", "статус")),
    ("бот: Вы юрлицо?", "no", ("отриц", "статус")),
    ("бот: Вы юрлицо?", "legal", ("подтверж", "юрид")),
    ("бот: По TIS подскажите бренды.", "все", ("все", "бренд")),
    ("бот: По TIS подскажите бренды.", "а какие есть", ("перечис", "бренд")),
    ("клиент: есть Volvo?\nбот: есть", "на него", ("ранее", "бренд")),
    ("клиент: есть Volvo?\nбот: есть", "для него", ("ранее", "бренд")),
    ("клиент: TIS", "TIS", ("корот", "tis")),
    ("клиент: EPC", "EPC", ("корот", "epc")),
    ("клиент: демо", "демо", ("демо", "доступ")),
    ("клиент: Почему не продаете физлицам?", "Почему не продаете физлицам?", ("причин", "физ")),
    ("клиент: Хочу купить как физ лицо", "Хочу купить как физ лицо", ("куп", "физ")),
    ("клиент: вы мошенники", "вы мошенники", ("обвин", "заявлен")),
    ("клиент: мне обещали бесплатно навсегда", "мне обещали бесплатно навсегда", ("обещал", "бесплат")),
]

_FALLBACK_CASES = [
    "PERSISTENT_INVALID_JSON_CASE",
]


for index, (dialog_text, user_query, expected_keywords) in enumerate(_INTEGRATION_CASES, start=1):
    setattr(
        ContextUnderstandingAgent01Tests,
        f"test_agent_integration_{index:02d}",
        _make_agent_keyword_test(dialog_text, user_query, expected_keywords),
    )

for index, user_query in enumerate(_FALLBACK_CASES, start=1):
    setattr(
        ContextUnderstandingAgent01Tests,
        f"test_agent_fallback_{index:02d}",
        _make_agent_fallback_test(user_query),
    )


class ContextUnderstandingAgentDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.prompt_path = cls.project_root / cls.config.paths.context_understanding_prompt

    def test_llm_exception_returns_fallback(self) -> None:
        agent = ContextUnderstandingAgent(_ScriptedContextLLM(raise_on_call=1), self.prompt_path)
        result = agent.understand("бот: Вы юрлицо?", "да")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "context_understanding_llm_error")

    def test_schema_retry_recovers_from_extra_field(self) -> None:
        llm = _ScriptedContextLLM()
        agent = ContextUnderstandingAgent(llm, self.prompt_path)
        result = agent.understand("бот: Вы юрлицо?", "RETRY_EXTRA_FIELD_CASE")
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.schema_retry_used)
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result.raw_response, llm.last_response)

    def test_retry_is_used_for_persistent_invalid_json_but_still_falls_back(self) -> None:
        llm = _ScriptedContextLLM()
        agent = ContextUnderstandingAgent(llm, self.prompt_path)
        result = agent.understand("бот: Вы юрлицо?", "PERSISTENT_INVALID_JSON_CASE")
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.schema_retry_used)
        self.assertEqual(result.fallback_reason, "json_object_not_found")
        self.assertEqual(llm.call_count, 3)

    def test_retry_recovers_from_empty_response(self) -> None:
        llm = _ScriptedContextLLM()
        agent = ContextUnderstandingAgent(llm, self.prompt_path)
        result = agent.understand("бот: Вы юрлицо?", "RETRY_EMPTY_RESPONSE_CASE")
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.schema_retry_used)
        self.assertEqual(llm.call_count, 2)

    def test_retry_recovers_from_json_object_not_found(self) -> None:
        llm = _ScriptedContextLLM()
        agent = ContextUnderstandingAgent(llm, self.prompt_path)
        result = agent.understand("бот: Вы юрлицо?", "RETRY_JSON_OBJECT_NOT_FOUND_CASE")
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.schema_retry_used)
        self.assertEqual(llm.call_count, 2)

    def test_retry_recovers_from_json_decode_error(self) -> None:
        llm = _ScriptedContextLLM()
        agent = ContextUnderstandingAgent(llm, self.prompt_path)
        result = agent.understand("бот: Вы юрлицо?", "RETRY_JSON_DECODE_CASE")
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.schema_retry_used)
        self.assertEqual(llm.call_count, 2)

    def test_retry_recovers_from_empty_required_field(self) -> None:
        llm = _ScriptedContextLLM()
        agent = ContextUnderstandingAgent(llm, self.prompt_path)
        result = agent.understand("бот: Вы юрлицо?", "RETRY_EMPTY_REQUIRED_FIELD_CASE")
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.schema_retry_used)
        self.assertEqual(llm.call_count, 2)


if __name__ == "__main__":
    unittest.main()
