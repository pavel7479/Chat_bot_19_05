from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.agents.intent_agent import IntentAgent
from src.config.loader import ConfigLoader
from src.core.interfaces import LLMProvider
from src.core.models import SessionState
from src.prompting.prompt_manager import PromptManager
from the_First_Agent.Agent_Zero.models import ContextUnderstandingResult
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.config.resource_paths import SEMANTIC_INTENTS_PATH, TOPIC_CLASSIFIER_PROMPT_PATH
from the_First_Agent.orchestrator.topic_classifier import TopicClassifier
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder


class _FakeAgentZero:
    def understand(self, dialog_text: str, user_query: str) -> ContextUnderstandingResult:
        return ContextUnderstandingResult(
            gist="Клиент уточняет наличие конкретных брендов.",
            meaning="Клиент спрашивает, есть ли в системе конкретные бренды Mercedec и Wolksvagen.",
            raw_response='{"gist":"x","meaning":"y"}',
            parsed_json={"gist": "x", "meaning": "y"},
        )


class _FakeBrandLLM(LLMProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return ""

    def generate_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "intent_1": {
                    "intent_id": "specific_brand_check",
                    "score": 0.96,
                    "reason": "Клиент спрашивает наличие конкретных брендов.",
                },
                "intent_2": None,
            },
            ensure_ascii=False,
        )


class E2EBrandTypo01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config = ConfigLoader(cls.project_root / "config.yaml").load()
        cls.topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
        cls.prompt_manager = PromptManager(
            cls.project_root,
            str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(cls.project_root)),
            cls.config.paths.answer_generator_prompt,
        )
        cls.topic_prompt_sections_builder = TopicPromptSectionsBuilder()

    def test_e2e_brand_typo_does_not_corrupt_examples_and_returns_specific_brand_check(self) -> None:
        llm = _FakeBrandLLM()
        classifier = TopicClassifier(
            llm_provider=llm,
            topic_ids=set(self.topic_catalog.topics.keys()),
            topic_titles_by_id=self.topic_catalog.title_map(),
            intents_config_path=self.project_root / self.config.paths.intents_config_file,
            brands_file_path=self.project_root / self.config.paths.brands_file,
        )
        agent = IntentAgent(
            topic_catalog=self.topic_catalog,
            topic_shortlist_builder=TopicShortlistBuilder(self.topic_catalog.topics, top_k=8),
            topic_classifier=classifier,
            prompt_manager=self.prompt_manager,
            topic_prompt_sections_builder=self.topic_prompt_sections_builder,
            context_understanding_agent=_FakeAgentZero(),
        )

        prompt, result = agent.classify(
            history_text="user: что умеет сервис",
            user_query="а Mercedec и Wolksvagen есть?",
            session_state=SessionState(),
        )

        self.assertNotIn(
            "Пример:\nДиалог:\nклиент: что умеет сервис\n\nПоследняя реплика клиента:\nа Mercedec и Wolksvagen есть?",
            prompt,
        )
        shortlist_ids = result.diagnostics.get("shortlist_topic_ids", [])
        self.assertIn("specific_brand_check", shortlist_ids)
        self.assertEqual(result.topic_ids[0], "specific_brand_check")


if __name__ == "__main__":
    unittest.main()
