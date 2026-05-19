from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.agents.policy.types import ResponseAction, ResponseState
from src.agents.response_planner import ResponsePlan, ResponsePlanner
from src.agents.response_policy import ResponseContractValidator
from src.agents.response_services.pipeline_services import (
    AnswerSynthesisService,
    GroundedAnswerComposer,
    ResponseModelFactory,
)
from src.core.models import RetrievedChunk, TopicClassificationResult


class _DummyPromptManager:
    def build_answer_prompt(
        self,
        dialog_text: str,
        rewritten_query: str,
        intents_with_scores: str,
        normalized_brands: str,
        retrieved_facts_text: str,
        user_query: str,
    ) -> str:
        return retrieved_facts_text


class _DummyLLM:
    def __init__(self, raw_response: str) -> None:
        self._raw_response = raw_response

    def generate_json(self, prompt: str) -> str:
        return self._raw_response


class _AlwaysFalseValidator:
    def validate(self, action: ResponseAction, answer_text: str) -> bool:
        return False


class _IdentityAntiRepeat:
    def apply(
        self,
        action: ResponseAction,
        answer_text: str,
        history_text: str,
        user_query: str,
    ) -> str:
        return answer_text


class _EmptyComposer:
    def compose(
        self,
        action_name: str,
        response_plan: list[str],
        chunks: list[RetrievedChunk],
        fallback_answer: str,
    ) -> str:
        return ""


class SecondAgentTransition01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.response_policy_path = cls.project_root / "src/config/response_policy.yaml"
        cls.fact_map_path = cls.project_root / "src/config/response_fact_map.yaml"

    def test_answer_sections_are_built_from_response_plan(self) -> None:
        factory = ResponseModelFactory()
        topic_result = TopicClassificationResult(
            topic_ids=["company_services_info"],
            confidence=0.91,
            reason="test",
        )
        plan = ResponsePlan(
            primary_action="company_services",
            primary_topic="company_services_info",
            required_fact_ids=["company_services_general", "company_services_competitor_discount"],
        )
        response = factory.build(
            topic_result=topic_result,
            selected_action=ResponseAction("company_services", "company_services_info"),
            response_plan=plan,
            effective_chunks=[],
            answer_text="Могу рассказать о возможностях сервиса для автобизнеса.",
            reasoning_summary="test",
        )
        self.assertEqual(
            response.answer_sections,
            ["company_services_general", "company_services_competitor_discount"],
        )

    def test_company_services_contract_uses_any_token_match(self) -> None:
        validator = ResponseContractValidator(self.response_policy_path)
        action = ResponseAction("company_services", "company_services_info")
        self.assertTrue(
            validator.validate(
                action,
                "Могу кратко рассказать о возможностях сервиса для автобизнеса.",
            )
        )

    def test_empty_answer_is_replaced_with_safe_default_after_validation(self) -> None:
        service = AnswerSynthesisService()
        topic_result = TopicClassificationResult(
            topic_ids=["company_services_info"],
            confidence=0.8,
            reason="test",
        )
        result = service.build_answer(
            llm_provider=_DummyLLM('{"answer_text":"мимо"}'),
            prompt_manager=_DummyPromptManager(),
            composer=_EmptyComposer(),
            validator=_AlwaysFalseValidator(),
            anti_repeat_policy=_IdentityAntiRepeat(),
            history_text="",
            user_query="что умеет сервис",
            topic_result=topic_result,
            response_state=ResponseState(),
            selected_action=ResponseAction("company_services", "company_services_info"),
            response_plan=ResponsePlan(
                primary_action="company_services",
                primary_topic="company_services_info",
            ),
            effective_chunks=[],
            retrieved_context="FACTS:\nСервис помогает автобизнесу работать с каталогами.",
            evidence_reason="knowledge_context",
            answerability_status="ok",
        )
        self.assertTrue(result.answer_text.strip())

    def test_prepared_context_has_priority_over_generic_chunks(self) -> None:
        service = AnswerSynthesisService()
        topic_result = TopicClassificationResult(
            topic_ids=["tis_tariffs"],
            confidence=0.8,
            reason="test",
        )
        result = service.build_answer(
            llm_provider=_DummyLLM('{"answer_text":"TIS Audi — 6000 руб."}'),
            prompt_manager=_DummyPromptManager(),
            composer=GroundedAnswerComposer(),
            validator=ResponseContractValidator(self.response_policy_path),
            anti_repeat_policy=_IdentityAntiRepeat(),
            history_text="",
            user_query="сколько стоит tis для audi",
            topic_result=topic_result,
            response_state=ResponseState(),
            selected_action=ResponseAction("tis_tariffs", "tis_tariffs"),
            response_plan=ResponsePlan(
                primary_action="tis_tariffs",
                primary_topic="tis_tariffs",
                required_price_blocks=["tis"],
            ),
            effective_chunks=[
                RetrievedChunk(
                    text="TIS подключается по брендам.",
                    score=1.0,
                    source="facts.yaml",
                    metadata={"section_tag": "general"},
                )
            ],
            retrieved_context="PRICES:\nTIS Audi — 6000",
            evidence_reason="knowledge_context",
            answerability_status="ok",
        )
        self.assertIn("TIS Audi — 6000", result.answer_prompt)

    def test_epc_and_tis_pricing_is_not_compare_action(self) -> None:
        planner = ResponsePlanner(self.fact_map_path)
        plan = planner.plan(
            topic_result=TopicClassificationResult(
                topic_ids=["epc_tariffs", "tis_tariffs"],
                confidence=0.84,
                reason="test",
            ),
            user_query="сколько стоит подписка",
            history_text="",
        )
        self.assertNotEqual(plan.primary_action, "compare_epc_tis")
        self.assertEqual(plan.required_price_blocks, ["epc", "tis"])

    def test_all_planner_actions_have_fact_map_default(self) -> None:
        raw = yaml.safe_load(self.fact_map_path.read_text(encoding="utf-8")) or {}
        self.assertIsInstance(raw, dict)
        missing: list[str] = []
        for action in set(ResponsePlanner._TOPIC_TO_ACTION.values()):
            variants = raw.get(action)
            if not isinstance(variants, dict) or "default" not in variants:
                missing.append(action)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
