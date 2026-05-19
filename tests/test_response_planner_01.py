from __future__ import annotations

import unittest
from pathlib import Path

from src.agents.response_planner import ResponsePlanner
from src.app.knowledge_retriever import KnowledgeRetriever
from src.core.models import TopicClassificationResult
from src.retrieval.fact_repository import FactRepository


class ResponsePlanner01Tests(unittest.TestCase):
    def setUp(self) -> None:
        project_root = Path('/root/project/Chat_bot')
        self.fact_repository = FactRepository(project_root / 'src/config/facts.yaml')
        self.planner = ResponsePlanner(project_root / 'src/config/response_fact_map.yaml')
        self.retriever = KnowledgeRetriever(fact_repository=self.fact_repository, max_facts_per_turn=8)

    def test_company_services_smoke_uses_only_overview_facts(self) -> None:
        topic_result = TopicClassificationResult(
            topic_ids=['company_services_info'],
            confidence=0.95,
            reason='service overview',
            diagnostics={},
        )
        plan = self.planner.plan(topic_result=topic_result, user_query='что умеет сервис')
        self.assertEqual(plan.primary_action, 'company_services')
        self.assertEqual(
            plan.required_fact_ids,
            ['company_services_general', 'company_services_competitor_discount'],
        )

        self.retriever.enrich(
            topic_result=topic_result,
            user_query='что умеет сервис',
            history_text='user: что умеет сервис',
            response_plan=plan,
        )
        retrieved = topic_result.diagnostics['first_agent_data']['knowledge']['retrieved_facts']
        fact_ids = [row['fact_id'] for row in retrieved]
        self.assertEqual(
            fact_ids,
            ['company_services_general', 'company_services_competitor_discount'],
        )
        forbidden = {
            'usage_limits_no_restrictions',
            'multi_user_access_policy',
            'multi_device_access_policy',
            'parts_selection_not_supported',
        }
        self.assertTrue(forbidden.isdisjoint(set(fact_ids)))


if __name__ == '__main__':
    unittest.main()
