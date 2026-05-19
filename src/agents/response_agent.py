from __future__ import annotations

from pathlib import Path

from src.agents.response_planner import ResponsePlan
from src.agents.response_planner import ResponsePlanner
from src.agents.policy.types import ResponseState
from src.agents.policy.types import ResponseAction
from src.agents.policy.anti_repeat_service import AntiRepeatService
from src.agents.policy.contract_validation_service import ContractValidationService
from src.agents.response_services.pipeline_services import (
    AnswerSynthesisService,
    GroundedAnswerComposer,
    ResponseModelFactory,
)
from src.core.interfaces import LLMProvider
from src.core.models import (
    BotResponse,
    RetrievedChunk,
    TopicClassificationResult,
)
from src.prompting.prompt_manager import PromptManager


class ResponseAgent:
    """Second agent (part 2): generate final manager-style answer from retrieved KB context."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        brands_file_path: Path,
        facts_file_path: Path,
        response_policy_file_path: Path,
        prices_file_path: Path,
        response_fact_map_file_path: Path,
        min_evidence_score: float = 0.15,
        min_evidence_hits: int = 1,
    ) -> None:
        self._llm = llm_provider
        self._prompt_manager = prompt_manager
        self._anti_repeat_policy = AntiRepeatService(response_policy_file_path)
        self._validator = ContractValidationService(response_policy_file_path)
        self._composer = GroundedAnswerComposer()
        self._answer_synthesis_service = AnswerSynthesisService()
        self._response_model_factory = ResponseModelFactory()
        self._planner = ResponsePlanner(
            response_fact_map_file_path,
            brands_file_path,
            facts_file_path=facts_file_path,
        )

    def plan(
        self,
        topic_result: TopicClassificationResult,
        user_query: str,
        history_text: str = "",
    ) -> ResponsePlan:
        return self._planner.plan(
            topic_result=topic_result,
            user_query=user_query,
            history_text=history_text,
        )

    def generate(
        self,
        history_text: str,
        user_query: str,
        topic_result: TopicClassificationResult,
        chunks: list[RetrievedChunk],
        retrieved_context: str,
        response_plan: ResponsePlan | None = None,
    ) -> tuple[str, str, BotResponse]:
        response_state = ResponseState.from_snapshot(topic_result.state_snapshot)
        plan = response_plan or self.plan(topic_result=topic_result, user_query=user_query, history_text=history_text)
        selected_action = ResponseAction(
            name=plan.primary_action,
            primary_topic=plan.primary_topic,
            secondary_topic=plan.secondary_topic,
            locked_action=bool(plan.locked_action),
        )

        prepared_context = self._extract_prepared_context(topic_result)
        effective_retrieved_context = prepared_context or retrieved_context
        effective_chunks = list(chunks)

        answer_result = self._answer_synthesis_service.build_answer(
            llm_provider=self._llm,
            prompt_manager=self._prompt_manager,
            composer=self._composer,
            validator=self._validator,
            anti_repeat_policy=self._anti_repeat_policy,
            history_text=history_text,
            user_query=user_query,
            topic_result=topic_result,
            response_state=response_state,
            selected_action=selected_action,
            response_plan=plan,
            effective_chunks=effective_chunks,
            retrieved_context=effective_retrieved_context,
            evidence_reason="knowledge_context",
            answerability_status="ok",
        )
        if isinstance(topic_result.diagnostics, dict):
            topic_result.diagnostics["answer_selection_trace"] = dict(answer_result.selection_trace)

        response = self._response_model_factory.build(
            topic_result=topic_result,
            selected_action=selected_action,
            response_plan=plan,
            effective_chunks=effective_chunks,
            answer_text=answer_result.answer_text,
            reasoning_summary=answer_result.reasoning_summary,
        )
        return answer_result.answer_prompt, answer_result.raw_answer, response

    @staticmethod
    def _extract_prepared_context(topic_result: TopicClassificationResult) -> str:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            return ""
        answer_block = first_agent_data.get("answer", {})
        if not isinstance(answer_block, dict):
            return ""
        return str(answer_block.get("prepared_context", "")).strip()
