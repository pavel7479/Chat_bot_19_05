from __future__ import annotations

from time import perf_counter
from pathlib import Path
from uuid import uuid4

from src.agents.intent_agent import IntentAgent
from src.agents.response_agent import ResponseAgent
from src.app.answer_composer import AnswerComposer
from src.app.brand_alias_resolver import TurnBrandAliasResolver
from src.app.dialog_act_router import DialogActRouter
from src.app.greeting_service import GreetingService
from src.app.knowledge_retriever import KnowledgeRetriever
from src.app.product_resolver import ProductResolver
from src.app.slot_extraction_service import SlotExtractionService
from src.app.state_update_service import StateUpdateService
from src.app.telemetry_service import TelemetryService
from src.config.schema import Config
from src.core.diff_utils import snapshot
from src.core.models import BotResponse, RetrievedChunk, SessionState
from src.session.session_manager import SessionManager
from the_First_Agent.context.history_formatter import format_history


class ChatBotOrchestrator:
    def __init__(
        self,
        config: Config,
        project_root: Path,
        intent_agent: IntentAgent,
        response_agent: ResponseAgent,
        knowledge_retriever: KnowledgeRetriever | None,
        brand_alias_resolver: TurnBrandAliasResolver | None,
        answer_composer: AnswerComposer | None,
        session_manager: SessionManager,
        logger,
        slot_extraction_service: SlotExtractionService | None = None,
        state_update_service: StateUpdateService | None = None,
        telemetry_service: TelemetryService | None = None,
        greeting_service: GreetingService | None = None,
        product_resolver: ProductResolver | None = None,
        dialog_act_router: DialogActRouter | None = None,
    ) -> None:
        self._config = config
        self._project_root = project_root
        self._intent_agent = intent_agent
        self._response_agent = response_agent
        self._knowledge_retriever = knowledge_retriever
        self._brand_alias_resolver = brand_alias_resolver
        self._answer_composer = answer_composer
        self._session = session_manager
        self._logger = logger
        self._slot_extractor = slot_extraction_service
        self._state_update = state_update_service or StateUpdateService()
        self._telemetry = telemetry_service or TelemetryService(logger)
        self._greeting_service = greeting_service or GreetingService()
        self._product_resolver = product_resolver
        self._dialog_act_router = dialog_act_router
        self._last_turn_debug: dict[str, dict[str, object]] = {}

    def respond(self, session_id: str, user_query: str) -> BotResponse:
        total_started = perf_counter()
        run_id = uuid4().hex
        self._session.add_user_message(session_id, user_query)
        history = self._session.get_history(session_id)
        turn_id = len([msg for msg in history if msg.role == "user"])
        session_state = self._session.get_state(session_id)
        history_text = format_history(history)

        classify_started = perf_counter()
        self._telemetry.topic_request(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            user_query=user_query,
            session_state=session_state.as_dict(),
        )
        slot_result = (
            self._slot_extractor.extract(user_query=user_query, session_state=session_state)
            if self._slot_extractor is not None
            else None
        )
        context_result = self._intent_agent.analyze_context(
            history_text=history_text,
            user_query=user_query,
        )
        product_context = (
            self._product_resolver.resolve(
                user_query=user_query,
                history_text=history_text,
                state_snapshot=session_state.as_dict(),
                slot_trace=dict(slot_result.trace) if slot_result is not None else {},
            )
            if self._product_resolver is not None
            else None
        )
        dialog_decision = (
            self._dialog_act_router.route(
                user_query=user_query,
                session_state=session_state,
                context_result=context_result,
                slot_trace=dict(slot_result.trace) if slot_result is not None else {},
                product_context=product_context,
            )
            if self._dialog_act_router is not None and product_context is not None
            else None
        )
        if dialog_decision is not None and dialog_decision.applied and dialog_decision.action_name == "greeting_once":
            topic_prompt, topic_result = self._intent_agent.build_from_dialog_act(
                history_text=history_text,
                user_query=user_query,
                decision=dialog_decision,
                context_result=context_result,
                slot_trace=dict(slot_result.trace) if slot_result is not None else {},
                product_context=product_context,
                session_state=session_state,
            )
            classify_ms = round((perf_counter() - classify_started) * 1000, 2)
            self._telemetry.topic_response(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                topic_prompt=topic_prompt,
                topic_result=topic_result,
                classify_ms=classify_ms,
            )
            self._telemetry.agent_zero_trace(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                trace=topic_result.diagnostics.get("agent_zero_trace", {}),
            )
            self._telemetry.classifier_routing(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                topic_result=topic_result,
            )
            self._telemetry.classifier_state_trace(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                topic_result=topic_result,
            )
            self._telemetry.classifier_quality(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                topic_result=topic_result,
            )
            response = self._greeting_service.build_short_circuit_bot_response()
            self._session.add_bot_message(session_id, response.answer_text)
            state_before_snapshot = session_state.as_dict()
            state_after_classification = self._state_update.apply_after_classification(
                base_state=session_state,
                topic_result=topic_result,
            )
            current_state = self._state_update.apply_after_response(
                state_before_response=state_after_classification,
                previous_state=session_state,
                response=response,
            )
            self._session.set_state(session_id, state=current_state)
            timings_ms = {
                "classify": classify_ms,
                "retrieve": 0.0,
                "generate": 0.0,
                "total": round((perf_counter() - total_started) * 1000, 2),
            }
            self._telemetry.turn_summary(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                user_query=user_query,
                topic_result=topic_result,
                response_plan={"primary_action": "greeting_once"},
                response=response,
                answer_block={},
                retrieval_trace={},
                state_before=state_before_snapshot,
                state_after_classification=state_after_classification.as_dict(),
                state_after_response=current_state.as_dict(),
                timings_ms=timings_ms,
            )
            self._telemetry.failure_event(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                user_query=user_query,
                topic_result=topic_result,
                response=response,
                answer_block={},
                state_before=state_before_snapshot,
                state_after_response=current_state.as_dict(),
            )
            self._last_turn_debug[session_id] = {
                "run_id": run_id,
                "turn_id": turn_id,
                "topic_prompt": topic_prompt,
                "topic_result_diagnostics": snapshot(topic_result.diagnostics),
                "state_before": state_before_snapshot,
                "state_after_classification": state_after_classification.as_dict(),
                "response_plan": {"primary_action": "greeting_once"},
                "retrieval_trace": {},
                "answer_prompt": "",
                "raw_answer": "",
                "response": {
                    "topic_id": response.topic_id,
                    "topic_ids": list(response.topic_ids),
                    "action_name": response.action_name,
                    "planned_action": response.planned_action,
                    "answer_text": response.answer_text,
                    "used_evidence_ids": list(response.used_evidence_ids),
                    "answer_sections": list(response.answer_sections),
                    "reasoning_summary": response.reasoning_summary,
                },
                "state_after_response": current_state.as_dict(),
            }
            return response
        if dialog_decision is not None and dialog_decision.applied and product_context is not None:
            topic_prompt, topic_result = self._intent_agent.build_from_dialog_act(
                history_text=history_text,
                user_query=user_query,
                decision=dialog_decision,
                context_result=context_result,
                slot_trace=dict(slot_result.trace) if slot_result is not None else {},
                product_context=product_context,
                session_state=session_state,
            )
        else:
            topic_prompt, topic_result = self._intent_agent.classify(
                history_text=history_text,
                user_query=user_query,
                session_state=session_state,
                slot_trace=dict(slot_result.trace) if slot_result is not None else {},
                context_result=context_result,
            )
        response_plan = self._response_agent.plan(
            topic_result=topic_result,
            user_query=user_query,
            history_text=history_text,
        )
        topic_result.diagnostics.setdefault("response_plan", response_plan.as_dict())
        if slot_result is not None:
            topic_result.diagnostics["slot_extraction_trace"] = dict(slot_result.trace)
            turn_analysis = topic_result.diagnostics.setdefault("turn_analysis", {})
            if isinstance(turn_analysis, dict):
                slots = turn_analysis.setdefault("slots", {})
                if isinstance(slots, dict):
                    slots.update(slot_result.slots)
        classify_ms = round((perf_counter() - classify_started) * 1000, 2)
        self._telemetry.topic_response(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_prompt=topic_prompt,
            topic_result=topic_result,
            classify_ms=classify_ms,
        )
        self._telemetry.agent_zero_trace(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            trace=topic_result.diagnostics.get("agent_zero_trace", {}),
        )
        self._telemetry.classifier_routing(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_result=topic_result,
        )
        self._telemetry.classifier_state_trace(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_result=topic_result,
        )
        self._telemetry.classifier_quality(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_result=topic_result,
        )
        if self._knowledge_retriever is not None:
            before_knowledge = self._extract_first_agent_block(topic_result, "knowledge")
            self._knowledge_retriever.enrich(
                topic_result=topic_result,
                user_query=user_query,
                history_text=history_text,
                response_plan=response_plan,
            )
            self._telemetry.first_agent_data_diff(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                actor="KnowledgeRetriever",
                step="knowledge_retriever",
                target="first_agent_data.knowledge",
                before_data=before_knowledge,
                after_data=self._extract_first_agent_block(topic_result, "knowledge"),
            )
        if self._brand_alias_resolver is not None:
            before_brands = self._extract_first_agent_block(topic_result, "brands")
            self._brand_alias_resolver.enrich(
                topic_result=topic_result,
                user_query=user_query,
            )
            self._telemetry.first_agent_data_diff(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                actor="TurnBrandAliasResolver",
                step="brand_alias_resolver",
                target="first_agent_data.brands",
                before_data=before_brands,
                after_data=self._extract_first_agent_block(topic_result, "brands"),
            )
        if self._answer_composer is not None:
            before_answer = self._extract_first_agent_block(topic_result, "answer")
            self._answer_composer.enrich(
                topic_result=topic_result,
                user_query=user_query,
                history_text=history_text,
                response_plan=response_plan,
            )
            self._telemetry.first_agent_data_diff(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                actor="AnswerComposer",
                step="answer_composer",
                target="first_agent_data.answer",
                before_data=before_answer,
                after_data=self._extract_first_agent_block(topic_result, "answer"),
            )
        self._telemetry.first_agent_data_snapshot(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_result=topic_result,
        )
        classification_state_before = session_state.as_dict()
        current_state = self._state_update.apply_after_classification(
            base_state=session_state,
            topic_result=topic_result,
        )
        state_after_classification = current_state.as_dict()
        self._telemetry.state_transition(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            actor="StateUpdateService",
            step="apply_after_classification",
            before_state=classification_state_before,
            after_state=state_after_classification,
        )
        self._session.set_state(session_id, state=current_state)

        retrieval_started = perf_counter()
        chunks, retrieved_context, retrieval_trace = self._build_knowledge_retrieval_payload(
            user_query=user_query,
            topic_result=topic_result,
            response_plan=response_plan,
        )
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000, 2)

        self._telemetry.retrieval_response(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_ids=topic_result.topic_ids,
            retrieval_ms=retrieval_ms,
            retrieval_trace=retrieval_trace,
            chunks=chunks,
        )
        topic_result.retrieval_context = dict(retrieval_trace)
        self._telemetry.pipeline_trace(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            retrieval_trace=retrieval_trace,
        )

        generate_started = perf_counter()
        self._telemetry.answer_request(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            topic_ids=topic_result.topic_ids,
            history_text=history_text,
            retrieved_context=retrieved_context,
            chunk_count=len(chunks),
        )
        answer_prompt, raw_answer, response = self._response_agent.generate(
            history_text=history_text,
            user_query=user_query,
            topic_result=topic_result,
            chunks=chunks,
            retrieved_context=retrieved_context,
            response_plan=response_plan,
        )
        if "answer_selection_trace" not in topic_result.diagnostics:
            topic_result.diagnostics["answer_selection_trace"] = {}
            self._logger.warning(
                "answer_selection_trace_missing_after_generate",
                extra={
                    "session_id": session_id,
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "topic_ids": list(topic_result.topic_ids),
                    "selected_action": response.action_name,
                },
            )
        response.answer_text = self._greeting_service.apply(
            answer_text=response.answer_text,
            state_before_response=session_state,
        )
        generate_ms = round((perf_counter() - generate_started) * 1000, 2)
        self._telemetry.answer_response(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            answer_prompt=answer_prompt,
            raw_answer=raw_answer,
            generate_ms=generate_ms,
        )
        self._telemetry.answer_selection_trace(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            trace=topic_result.diagnostics.get("answer_selection_trace", {}),
        )

        self._session.add_bot_message(session_id, response.answer_text)
        response_state_before = current_state.as_dict()
        current_state = self._state_update.apply_after_response(
            state_before_response=current_state,
            previous_state=session_state,
            response=response,
            answer_block=self._extract_first_agent_block(topic_result, "answer"),
        )
        self._telemetry.state_transition(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            actor="StateUpdateService",
            step="apply_after_response",
            before_state=response_state_before,
            after_state=current_state.as_dict(),
        )
        self._session.set_state(session_id, state=current_state)
        total_ms = round((perf_counter() - total_started) * 1000, 2)

        self._telemetry.request_telemetry(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            classify_ms=classify_ms,
            retrieval_ms=retrieval_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
        )

        self._telemetry.final_decision(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            response=response,
            classify_ms=classify_ms,
            retrieval_ms=retrieval_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
        )
        answer_block = self._extract_first_agent_block(topic_result, "answer")
        timings_ms = {
            "classify": classify_ms,
            "retrieve": retrieval_ms,
            "generate": generate_ms,
            "total": total_ms,
        }
        self._telemetry.turn_summary(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            user_query=user_query,
            topic_result=topic_result,
            response_plan=response_plan.as_dict(),
            response=response,
            answer_block=answer_block,
            retrieval_trace=retrieval_trace,
            state_before=classification_state_before,
            state_after_classification=state_after_classification,
            state_after_response=current_state.as_dict(),
            timings_ms=timings_ms,
        )
        self._telemetry.failure_event(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            user_query=user_query,
            topic_result=topic_result,
            response=response,
            answer_block=answer_block,
            state_before=classification_state_before,
            state_after_response=current_state.as_dict(),
        )
        self._last_turn_debug[session_id] = {
            "run_id": run_id,
            "turn_id": turn_id,
            "topic_prompt": topic_prompt,
            "topic_result_diagnostics": snapshot(topic_result.diagnostics),
            "first_agent_data": self._extract_first_agent_data(topic_result),
            "answer_block": answer_block,
            "response_plan": response_plan.as_dict(),
            "retrieval_trace": snapshot(retrieval_trace),
            "answer_prompt": answer_prompt,
            "raw_answer": raw_answer,
            "state_before": classification_state_before,
            "state_after_classification": state_after_classification,
            "response": {
                "topic_id": response.topic_id,
                "topic_ids": list(response.topic_ids),
                "action_name": response.action_name,
                "planned_action": response.planned_action,
                "answer_text": response.answer_text,
                "used_evidence_ids": list(response.used_evidence_ids),
                "answer_sections": list(response.answer_sections),
                "reasoning_summary": response.reasoning_summary,
            },
            "state_after_response": current_state.as_dict(),
        }

        return response

    def clear_session(self, session_id: str) -> None:
        self._session.clear(session_id)
        self._last_turn_debug.pop(session_id, None)

    def get_debug_trace(self, session_id: str) -> dict[str, object]:
        return dict(self._last_turn_debug.get(session_id, {}))

    @staticmethod
    def _build_knowledge_retrieval_payload(
        user_query: str,
        topic_result,
        response_plan=None,
    ) -> tuple[list[RetrievedChunk], str, dict[str, object]]:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            first_agent_data = {}
        knowledge = first_agent_data.get("knowledge", {})
        answer_block = first_agent_data.get("answer", {})
        if not isinstance(knowledge, dict):
            knowledge = {}
        if not isinstance(answer_block, dict):
            answer_block = {}

        retrieved_facts = knowledge.get("retrieved_facts", [])
        if not isinstance(retrieved_facts, list):
            retrieved_facts = []
        prepared_context = str(answer_block.get("prepared_context", "")).strip()
        structured_context = answer_block.get("structured_context", {})
        if not isinstance(structured_context, dict):
            structured_context = {}

        chunks: list[RetrievedChunk] = []
        selected_fact_ids: list[str] = []
        for fact in retrieved_facts:
            if not isinstance(fact, dict):
                continue
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            entry_index = str(fact.get("fact_id", "")).strip()
            section_tag = str(fact.get("section_tag", "general")).strip() or "general"
            priority_raw = fact.get("priority", 0)
            try:
                score = float(priority_raw)
            except (TypeError, ValueError):
                score = 0.0
            chunk = RetrievedChunk(
                text=text,
                score=score,
                source="facts.yaml",
                metadata={
                    "entry_index": entry_index,
                    "fact_id": entry_index,
                    "search_kind": "knowledge_context",
                    "section_tag": section_tag,
                    "why_selected": f"topic_match={fact.get('topic', '')}",
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                },
            )
            chunks.append(chunk)
            if entry_index:
                selected_fact_ids.append(entry_index)

        price_lines = structured_context.get("prices", [])
        if isinstance(price_lines, list):
            for idx, line in enumerate(price_lines):
                text = str(line).strip()
                if not text:
                    continue
                chunks.append(
                    RetrievedChunk(
                        text=text,
                        score=10.0,
                        source="prices.yaml",
                        metadata={
                            "entry_index": f"price:{idx}",
                            "fact_id": f"price:{idx}",
                            "search_kind": "prepared_context",
                            "section_tag": "pricing",
                            "why_selected": "response_plan_price_block",
                            "bm25_score": 0.0,
                            "dense_score": 0.0,
                        },
                    )
                )

        retrieval_trace = {
            "trace_id": uuid4().hex,
            "selection_mode": "deterministic_fact_ids",
            "query_received": {
                "query": user_query,
                "topic_ids": list(topic_result.topic_ids),
                "planned_action": response_plan.primary_action if response_plan is not None else topic_result.planned_action,
                "current_focus": topic_result.current_focus,
                "slots": dict(topic_result.state_snapshot.get("slots", {}))
                if isinstance(topic_result.state_snapshot, dict)
                and isinstance(topic_result.state_snapshot.get("slots", {}), dict)
                else {},
            },
            "selected_fact_ids": selected_fact_ids,
            "price_blocks": list(response_plan.required_price_blocks) if response_plan is not None else [],
            "chunks": [
                {
                    "score": round(chunk.score, 4),
                    "source": chunk.source,
                    "entry_index": chunk.metadata.get("entry_index", ""),
                    "fact_id": chunk.metadata.get("fact_id", ""),
                    "why_selected": str(chunk.metadata.get("why_selected", "")),
                    "section_tag": chunk.metadata.get("section_tag", "general"),
                }
                for chunk in chunks
            ],
        }
        return chunks, prepared_context, retrieval_trace

    @staticmethod
    def _extract_first_agent_block(topic_result, block_name: str) -> object:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            return {}
        return snapshot(first_agent_data.get(block_name, {}))

    @staticmethod
    def _extract_first_agent_data(topic_result) -> dict[str, object]:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            return {}
        return snapshot(first_agent_data)
