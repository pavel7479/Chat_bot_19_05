from __future__ import annotations

from typing import Any

from src.core.diff_utils import dict_diff, snapshot
from src.core.models import BotResponse, TopicClassificationResult
from src.logging_system.logger import StructuredLoggerFactory, log_event


class TelemetryService:
    """Centralized telemetry/trace logging for one chatbot turn."""

    def __init__(self, logger) -> None:
        self._logger = logger
        self._turn_logger = StructuredLoggerFactory.get_related_logger(logger.name, "turns")
        self._failure_logger = StructuredLoggerFactory.get_related_logger(logger.name, "failures")

    def topic_request(self, session_id: str, run_id: str, turn_id: int, user_query: str, session_state: dict[str, object]) -> None:
        log_event(
            self._logger,
            "topic_classification_request",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            user_query=user_query,
            session_state=session_state,
        )

    def topic_response(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        topic_prompt: str,
        topic_result: TopicClassificationResult,
        classify_ms: float,
    ) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        log_event(
            self._logger,
            "topic_classification_response",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            prompt=topic_prompt,
            topic_ids=topic_result.topic_ids,
            primary_topic_id=topic_result.primary_topic_id,
            confidence=topic_result.confidence,
            reason=topic_result.reason,
            classifier_source=topic_result.classifier_source,
            fallback_reason=topic_result.fallback_reason,
            pipeline_version=diagnostics.get("pipeline_version", ""),
            active_pipeline=diagnostics.get("active_pipeline", []),
            pipeline_contract=diagnostics.get("pipeline_contract", {}),
            raw_llm_response=diagnostics.get("raw_llm_response", ""),
            parsed_json=diagnostics.get("parsed_json", {}),
            normalization_trace=diagnostics.get("normalization_trace", {}),
            rewrite_trace=diagnostics.get("rewrite_trace", {}),
            validation_errors=diagnostics.get("validation_errors", []),
            agent_zero_trace=diagnostics.get("agent_zero_trace", {}),
            shortlist_trace=diagnostics.get("shortlist_trace", {}),
            followup_trace=diagnostics.get("followup_trace", {}),
            slot_extraction_trace=diagnostics.get("slot_extraction_trace", {}),
            state_before=diagnostics.get("state_before", {}),
            state_after=topic_result.state_snapshot,
            state_diff=diagnostics.get("state_diff", []),
            rule_trace=topic_result.rule_trace,
            duration_ms=classify_ms,
        )

    def agent_zero_trace(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        trace: dict[str, object],
    ) -> None:
        log_event(
            self._logger,
            "agent_zero_trace",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            trace=trace,
        )

    def classifier_routing(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        topic_result: TopicClassificationResult,
    ) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        log_event(
            self._logger,
            "classifier_routing",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            classifier_source=topic_result.classifier_source,
            fallback_reason=topic_result.fallback_reason,
            routing_mode="direct_json_pipeline",
            active_pipeline=diagnostics.get("active_pipeline", []),
            pipeline_version=diagnostics.get("pipeline_version", ""),
            pipeline_contract=diagnostics.get("pipeline_contract", {}),
            final_topics=topic_result.topic_ids,
            final_planned_action=topic_result.planned_action,
        )

    def classifier_state_trace(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        topic_result: TopicClassificationResult,
    ) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        trace = diagnostics.get("state_trace", [])
        if not isinstance(trace, list):
            trace = []
        log_event(
            self._logger,
            "classifier_state_trace",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            classifier_source=topic_result.classifier_source,
            steps=trace,
        )

    def classifier_quality(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        topic_result: TopicClassificationResult,
    ) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        trace = diagnostics.get("state_trace", [])
        trace_steps = []
        if isinstance(trace, list):
            trace_steps = [str(item.get("step", "")) for item in trace if isinstance(item, dict)]
        active_pipeline = diagnostics.get("active_pipeline", [])
        if not isinstance(active_pipeline, list):
            active_pipeline = []
        expected_steps = set(str(item) for item in active_pipeline if str(item).strip())
        observed_steps = set(trace_steps)
        trace_complete = expected_steps.issubset(observed_steps)

        log_event(
            self._logger,
            "classifier_quality",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            classifier_mode="direct_json_pipeline",
            classifier_source=topic_result.classifier_source,
            fallback_reason=topic_result.fallback_reason,
            planned_action=topic_result.planned_action,
            current_focus=topic_result.current_focus,
            topic_ids=list(topic_result.topic_ids),
            confidence=topic_result.confidence,
            pipeline_version=str(diagnostics.get("pipeline_version", "")),
            trace_complete=trace_complete,
            missing_steps=sorted(expected_steps.difference(observed_steps)),
            active_pipeline=active_pipeline,
            post_model_topic_adjustments_enabled=bool(
                isinstance(diagnostics.get("pipeline_contract", {}), dict)
                and diagnostics.get("pipeline_contract", {}).get("post_model_topic_adjustments_enabled", False)
            ),
        )

    def first_agent_data_snapshot(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        topic_result: TopicClassificationResult,
    ) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_output = diagnostics.get("first_agent_output", {})
        if not isinstance(first_agent_output, dict):
            first_agent_output = {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            first_agent_data = {}
        first_agent_trace = diagnostics.get("first_agent_trace", {})
        if not isinstance(first_agent_trace, dict):
            first_agent_trace = {}
        log_event(
            self._logger,
            "first_agent_data_snapshot",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            first_agent_output=first_agent_output,
            first_agent_data=first_agent_data,
            first_agent_trace=first_agent_trace,
        )

    def first_agent_data_diff(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        actor: str,
        step: str,
        target: str,
        before_data: object,
        after_data: object,
    ) -> None:
        before_snapshot = snapshot(before_data)
        after_snapshot = snapshot(after_data)
        log_event(
            self._logger,
            "first_agent_data_diff",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            actor=actor,
            step=step,
            target=target,
            before=before_snapshot,
            after=after_snapshot,
            diff=dict_diff(before_snapshot, after_snapshot),
        )

    def state_transition(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        actor: str,
        step: str,
        before_state: dict[str, object],
        after_state: dict[str, object],
    ) -> None:
        before_snapshot = snapshot(before_state)
        after_snapshot = snapshot(after_state)
        log_event(
            self._logger,
            "state_transition",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            actor=actor,
            step=step,
            before=before_snapshot,
            after=after_snapshot,
            diff=dict_diff(before_snapshot, after_snapshot),
        )

    def retrieval_response(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        topic_ids: list[str],
        retrieval_ms: float,
        retrieval_trace: dict[str, object],
        chunks: list[object],
    ) -> None:
        log_event(
            self._logger,
            "retrieval_response",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_ids=topic_ids,
            duration_ms=retrieval_ms,
            retrieval_trace=retrieval_trace,
            retrieval_hits=[
                {
                    "score": round(chunk.score, 3),
                    "source": chunk.source,
                    "entry_index": chunk.metadata.get("entry_index", ""),
                    "text_preview": chunk.text[:220],
                }
                for chunk in chunks
            ],
        )

    def pipeline_trace(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        retrieval_trace: dict[str, object],
    ) -> None:
        log_event(
            self._logger,
            "pipeline_trace",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            trace=retrieval_trace,
        )

    def answer_request(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        topic_ids: list[str],
        history_text: str,
        retrieved_context: str,
        chunk_count: int,
    ) -> None:
        log_event(
            self._logger,
            "llm_answer_request",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_ids=topic_ids,
            history_chars=len(history_text),
            retrieved_context_chars=len(retrieved_context),
            retrieval_hits_count=chunk_count,
        )

    def answer_response(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        answer_prompt: str,
        raw_answer: str,
        generate_ms: float,
    ) -> None:
        log_event(
            self._logger,
            "llm_answer_response",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            prompt=answer_prompt,
            raw_answer=raw_answer,
            duration_ms=generate_ms,
        )

    def answer_selection_trace(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        trace: dict[str, object],
    ) -> None:
        log_event(
            self._logger,
            "answer_selection_trace",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            trace=trace,
        )

    def request_telemetry(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        classify_ms: float,
        retrieval_ms: float,
        generate_ms: float,
        total_ms: float,
    ) -> None:
        log_event(
            self._logger,
            "request_telemetry",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            timings_ms={
                "classify": classify_ms,
                "retrieve": retrieval_ms,
                "generate": generate_ms,
                "total": total_ms,
            },
        )

    def final_decision(
        self,
        session_id: str,
        run_id: str,
        turn_id: int,
        response: BotResponse,
        classify_ms: float,
        retrieval_ms: float,
        generate_ms: float,
        total_ms: float,
    ) -> None:
        log_event(
            self._logger,
            "final_decision",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            chosen_topic=response.topic_id,
            chosen_topics=response.topic_ids,
            reasoning_summary=response.reasoning_summary,
            response_text=response.answer_text,
            media_count=len(response.media_refs),
            trace={
                "action": response.action_name,
                "planned_action": response.planned_action,
                "sections": list(response.answer_sections),
                "evidence_ids": list(response.used_evidence_ids),
                "contract_flags": dict(response.contract_flags),
            },
            timings_ms={
                "classify": classify_ms,
                "retrieve": retrieval_ms,
                "generate": generate_ms,
                "total": total_ms,
            },
        )

    def turn_summary(
        self,
        *,
        session_id: str,
        run_id: str,
        turn_id: int,
        user_query: str,
        topic_result: TopicClassificationResult,
        response_plan: dict[str, object],
        response: BotResponse,
        answer_block: dict[str, object],
        retrieval_trace: dict[str, object],
        state_before: dict[str, object],
        state_after_classification: dict[str, object],
        state_after_response: dict[str, object],
        timings_ms: dict[str, float],
    ) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        answer_selection = diagnostics.get("answer_selection_trace", {})
        topic_diagnostics = {
            "classifier_source": topic_result.classifier_source,
            "reason": topic_result.reason,
            "agent_zero_trace": diagnostics.get("agent_zero_trace", {}),
            "shortlist_trace": diagnostics.get("shortlist_trace", {}),
            "followup_trace": diagnostics.get("followup_trace", {}),
            "slot_extraction_trace": diagnostics.get("slot_extraction_trace", {}),
            "dialog_act_trace": diagnostics.get("dialog_act_trace", {}),
            "product_context_trace": diagnostics.get("product_context_trace", {}),
            "planner_trace": diagnostics.get("planner_trace", {}),
            "pricing_flow_trace": diagnostics.get("pricing_flow_trace", {}),
            "answer_selection_trace": answer_selection,
            "conversation_reasoning_trace": diagnostics.get("conversation_reasoning_trace", {}),
        }
        log_event(
            self._turn_logger,
            "turn_summary",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            user_query=user_query,
            topic_ids=list(topic_result.topic_ids),
            primary_topic_id=topic_result.primary_topic_id,
            selected_action=response.action_name,
            planned_action=response.planned_action,
            response_plan=response_plan,
            answer_text=response.answer_text,
            answer_source=str(answer_selection.get("final_answer_source", "")),
            answer_block=snapshot(answer_block),
            retrieval_trace=snapshot(retrieval_trace),
            state_before=snapshot(state_before),
            state_after_classification=snapshot(state_after_classification),
            state_after_response=snapshot(state_after_response),
            diagnostics=topic_diagnostics,
            timings_ms=timings_ms,
        )

    def failure_event(
        self,
        *,
        session_id: str,
        run_id: str,
        turn_id: int,
        user_query: str,
        topic_result: TopicClassificationResult,
        response: BotResponse,
        answer_block: dict[str, object],
        state_before: dict[str, object],
        state_after_response: dict[str, object],
    ) -> None:
        failure_flags = self._detect_failure_flags(
            topic_result=topic_result,
            response=response,
            answer_block=answer_block,
            state_before=state_before,
            state_after_response=state_after_response,
        )
        if not failure_flags:
            return
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        log_event(
            self._failure_logger,
            "turn_failure",
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            user_query=user_query,
            topic_ids=list(topic_result.topic_ids),
            selected_action=response.action_name,
            failure_flags=failure_flags,
            answer_text=response.answer_text,
            answer_block=snapshot(answer_block),
            answer_selection_trace=snapshot(diagnostics.get("answer_selection_trace", {})),
            state_before=snapshot(state_before),
            state_after_response=snapshot(state_after_response),
        )

    def _detect_failure_flags(
        self,
        *,
        topic_result: TopicClassificationResult,
        response: BotResponse,
        answer_block: dict[str, object],
        state_before: dict[str, object],
        state_after_response: dict[str, object],
    ) -> list[dict[str, object]]:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        answer_selection = diagnostics.get("answer_selection_trace", {})
        flags: list[dict[str, object]] = []

        if response.action_name == "pricing_summary" and str(state_after_response.get("active_pricing_flow", "")).strip() == "none":
            flags.append(
                {
                    "code": "pricing_flow_not_activated",
                    "message": "pricing_summary completed but active_pricing_flow stayed none",
                }
            )

        final_answer_source = str(answer_selection.get("final_answer_source", "")).strip().lower()
        if answer_selection.get("action_compatible") is False and final_answer_source == "llm":
            flags.append(
                {
                    "code": "llm_action_mismatch_selected",
                    "message": "LLM answer was selected despite action incompatibility",
                }
            )

        if response.action_name in {
            "human_operator",
            "ask_legal_status",
            "brand_group_clarification",
            "partial_catalog_restriction",
        } and final_answer_source == "llm":
            flags.append(
                {
                    "code": "locked_action_owned_by_llm",
                    "message": "Locked business action resolved to llm final answer",
                }
            )

        flags.extend(
            self._detect_pricing_flow_flags(
                response=response,
                answer_block=answer_block,
                state_after_response=state_after_response,
            )
        )
        return flags

    def _detect_pricing_flow_flags(
        self,
        *,
        response: BotResponse,
        answer_block: dict[str, object],
        state_after_response: dict[str, object],
    ) -> list[dict[str, object]]:
        if response.action_name != "tis_tariffs":
            return []

        flags: list[dict[str, object]] = []
        price_context = answer_block.get("price_context", {}) if isinstance(answer_block.get("price_context", {}), dict) else {}
        pricing_flow = state_after_response.get("pricing_flow", {}) if isinstance(state_after_response.get("pricing_flow", {}), dict) else {}
        brand_mentions = pricing_flow.get("brand_mentions", [])
        missing_brands = {
            str(item).strip().lower()
            for item in state_after_response.get("missing_price_brands", [])
            if str(item).strip()
        }
        display_by_canonical: dict[str, str] = {}
        if isinstance(brand_mentions, list):
            for item in brand_mentions:
                if not isinstance(item, dict):
                    continue
                canonical = str(item.get("canonical_brand", "")).strip().lower()
                display_name = str(item.get("display_name", "")).strip()
                if canonical and display_name and canonical not in display_by_canonical:
                    display_by_canonical[canonical] = display_name
        pricing_mode = str(state_after_response.get("pricing_mode", "")).strip()
        answer_text = str(response.answer_text or "")
        if pricing_mode == "remaining_only":
            priced_brands = {
                str(item).strip().lower()
                for item in state_after_response.get("priced_brands", [])
                if str(item).strip()
            }
            repeated_priced = []
            for canonical in sorted(priced_brands):
                display_name = display_by_canonical.get(canonical, canonical)
                if display_name and display_name.lower() in answer_text.lower():
                    repeated_priced.append(display_name)
            if repeated_priced:
                flags.append(
                    {
                        "code": "remaining_repeated_priced_brands",
                        "message": "remaining_only answer repeated already processed brands",
                        "brands": repeated_priced,
                    }
                )
            missing_in_answer = []
            for canonical in sorted(missing_brands):
                display_name = display_by_canonical.get(canonical, canonical)
                if display_name and display_name.lower() not in answer_text.lower():
                    missing_in_answer.append(display_name)
            if missing_in_answer:
                flags.append(
                    {
                        "code": "remaining_missing_brands_not_rendered",
                        "message": "remaining_only answer dropped recognized brands with missing price",
                        "brands": missing_in_answer,
                    }
                )

        if pricing_mode == "explain_unresolved" and "уточните стоимость" in answer_text.lower():
            flags.append(
                {
                    "code": "clarification_cost_loop",
                    "message": "clarification pushback answer loops back to asking cost clarification",
                }
            )

        unresolved_lines = answer_text.lower().splitlines()
        recognized_displays = {display.lower() for display in display_by_canonical.values() if display}
        for line in unresolved_lines:
            if "не распознан как бренд" not in line:
                continue
            for display in recognized_displays:
                if display and display in line:
                    flags.append(
                        {
                            "code": "recognized_brand_reported_as_unknown",
                            "message": "recognized brand was rendered as unknown in pricing clarification",
                            "line": line,
                        }
                    )
                    break

        if not price_context and response.action_name == "tis_tariffs":
            flags.append(
                {
                    "code": "missing_price_context_trace",
                    "message": "tis_tariffs response does not carry structured price_context",
                }
            )
        return flags
