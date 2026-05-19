from __future__ import annotations

import os
import unittest
from pathlib import Path

import requests

from src.config.loader import ConfigLoader
from tests.support.reliable_ollama_provider import ReliableOllamaTestProvider, TestLLMRuntimeConfig
from the_First_Agent.Agent_Zero.context_understanding_agent import ContextUnderstandingAgent


class ContextUnderstandingAgentRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.getenv("RUN_REAL_CONTEXT_UNDERSTANDING_TESTS") != "1":
            raise unittest.SkipTest("Real Agent Zero tests are disabled. Set RUN_REAL_CONTEXT_UNDERSTANDING_TESTS=1")

        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.prompt_path = cls.project_root / cls.config.paths.context_understanding_prompt
        cls._ensure_ollama_available(cls.config.llm.base_url)
        cls.llm = ReliableOllamaTestProvider(
            llm_config=cls.config.llm,
            runtime=TestLLMRuntimeConfig(per_request_timeout_s=35, max_retries=1, retry_backoff_s=0.5),
        )
        cls.agent = ContextUnderstandingAgent(cls.llm, cls.prompt_path)

    @staticmethod
    def _ensure_ollama_available(base_url: str) -> None:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5, proxies={"http": None, "https": None})
            response.raise_for_status()
        except Exception as error:
            raise unittest.SkipTest(f"Ollama is unavailable: {error}")

    def _assert_basic_contract(self, result) -> None:
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.gist.strip())
        self.assertTrue(result.meaning.strip())
        self.assertFalse(result.json_extracted_from_wrapped_response)
        lowered = f"{result.gist} {result.meaning}".lower()
        for forbidden in (
            "intent_id",
            "demo_access",
            "legal_entity_purchase_flow",
            "brand_list_request",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_real_yes_legal(self) -> None:
        result = self.agent.understand(
            dialog_text="бот: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?\nклиент: Да, являюсь",
            user_query="Да, являюсь",
        )
        self._assert_basic_contract(result)

    def test_real_switch_to_tis(self) -> None:
        result = self.agent.understand(
            dialog_text="клиент: Можно только один бренд?\nбот: EPC Full продается только полным пакетом.\nклиент: Либо тогда только TIS",
            user_query="Либо тогда только TIS",
        )
        self._assert_basic_contract(result)

    def test_real_noise_reply(self) -> None:
        result = self.agent.understand(
            dialog_text="бот: Мы можем предоставить демо только юрлицам. Вы представитель автобизнеса?\nклиент: ....548",
            user_query="....548",
        )
        self._assert_basic_contract(result)
        lowered = result.meaning.lower()
        self.assertTrue(any(token in lowered for token in ("шум", "нерелевант", "непонят", "не отвечает")))


if __name__ == "__main__":
    unittest.main()
