from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.app.knowledge_retriever import KnowledgeRetriever
from src.core.models import TopicClassificationResult
from src.domain.pricing import PriceCatalog
from src.retrieval.fact_repository import FactRepository


class SemanticSync01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.semantic_intents_path = cls.project_root / "the_First_Agent/config/semantic_intents.yaml"
        cls.facts_path = cls.project_root / "src/config/facts.yaml"
        cls.prices_path = cls.project_root / "src/config/prices.yaml"

    def test_semantic_intents_points_are_strings(self) -> None:
        raw = yaml.safe_load(self.semantic_intents_path.read_text(encoding="utf-8")) or {}
        intents = raw.get("intents", [])
        self.assertIsInstance(intents, list)
        for intent in intents:
            self.assertIsInstance(intent, dict)
            for field in ("choose_when", "not_choose_when", "examples"):
                value = intent.get(field)
                if isinstance(value, list):
                    self.assertTrue(
                        all(isinstance(item, str) for item in value),
                        msg=f"{intent.get('intent')} -> {field} contains non-string items: {value!r}",
                    )

    def test_knowledge_retriever_collects_facts_for_both_topics(self) -> None:
        retriever = KnowledgeRetriever(
            fact_repository=FactRepository(self.facts_path),
            price_catalog=PriceCatalog(self.prices_path),
        )
        topic_result = TopicClassificationResult(
            topic_ids=["demo_access", "legal_entity_purchase_flow"],
            confidence=0.9,
            reason="test",
            diagnostics={},
            state_snapshot={},
        )

        retriever.enrich(
            topic_result=topic_result,
            user_query="являюсь",
            history_text="бот: Демо только юрлицам. Вы представитель автобизнеса?",
        )

        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        self.assertIsInstance(first_agent_data, dict)
        knowledge = first_agent_data.get("knowledge", {})
        self.assertIsInstance(knowledge, dict)
        retrieved_facts = knowledge.get("retrieved_facts", [])
        self.assertIsInstance(retrieved_facts, list)

        retrieved_topics = {
            str(item.get("topic", "")).strip()
            for item in retrieved_facts
            if isinstance(item, dict) and str(item.get("topic", "")).strip()
        }
        self.assertIn("demo_access", retrieved_topics)
        self.assertIn("legal_entity_purchase_flow", retrieved_topics)


if __name__ == "__main__":
    unittest.main()
