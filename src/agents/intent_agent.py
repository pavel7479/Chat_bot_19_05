from __future__ import annotations

import json
import re
from pathlib import Path

from src.core.diff_utils import dict_diff
from src.core.models import ContextSignals, SessionState, TopicClassificationResult
from src.intents.semantic_compatibility_validator import SemanticCompatibilityValidator
from src.prompting.prompt_manager import PromptManager
from the_First_Agent.Agent_Zero.context_understanding_agent import ContextUnderstandingAgent
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.context.context_signal_extractor import ContextSignalExtractor
from the_First_Agent.followup.followup_resolver import FollowupResolver
from the_First_Agent.orchestrator.topic_classifier import TopicClassifier
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder


class IntentAgent:
    """First agent: classify user intent/topics with session-aware context."""

    _PARTIAL_EPC_PATTERNS = (
        re.compile(r"\bтолько одного бренда\b"),
        re.compile(r"\bпредоставить каталоги только одного бренда\b"),
        re.compile(r"\bне полный пакет\b"),
        re.compile(r"\bчасть каталога\b"),
        re.compile(r"\bотдельно по брендам\b"),
        re.compile(r"\bодин бренд в (?:epc|епс)\b"),
    )

    def __init__(
        self,
        topic_catalog: TopicCatalog,
        topic_shortlist_builder: TopicShortlistBuilder,
        topic_classifier: TopicClassifier,
        prompt_manager: PromptManager,
        topic_prompt_sections_builder: TopicPromptSectionsBuilder,
        context_understanding_agent: ContextUnderstandingAgent | None = None,
        context_signal_extractor: ContextSignalExtractor | None = None,
        followup_resolver: FollowupResolver | None = None,
    ) -> None:
        self._topic_catalog = topic_catalog
        self._topic_shortlist_builder = topic_shortlist_builder
        self._topic_classifier = topic_classifier
        self._prompt_manager = prompt_manager
        self._topic_prompt_sections_builder = topic_prompt_sections_builder
        self._context_understanding_agent = context_understanding_agent
        self._context_signal_extractor = context_signal_extractor
        self._followup_resolver = followup_resolver
        project_root = Path(__file__).resolve().parents[2]
        self._semantic_compatibility_validator = SemanticCompatibilityValidator(
            project_root / "src/config/semantic_topic_map.yaml"
        )

    def classify(
        self,
        history_text: str,
        user_query: str,
        session_state: SessionState,
        slot_trace: dict[str, object] | None = None,
        context_result=None,
    ) -> tuple[str, TopicClassificationResult]:
        slot_trace = slot_trace if isinstance(slot_trace, dict) else {}
        context_result = context_result or self._understand_context(history_text=history_text, user_query=user_query)
        semantic_frame = (
            context_result.semantic_frame.as_dict()
            if getattr(context_result, "semantic_frame", None) is not None
            else {}
        )
        context_signals, followup_resolution, followup_prompt_text = self._resolve_followup(
            history_text=history_text,
            user_query=user_query,
            session_state=session_state,
            slot_trace=slot_trace,
            context_result=context_result,
        )
        shortlist = self._topic_shortlist_builder.build_shortlist(
            user_query,
            history_text=history_text,
            session_state=session_state,
            context_signals=context_signals,
        )
        shortlist_ids = [item.topic_id for item in shortlist]
        prompt = self._prompt_manager.build_topic_prompt(
            allowed_intents_text=self._topic_catalog.allowed_intents_text(shortlist_ids),
            topics_text=self._topic_catalog.as_prompt_text(shortlist_ids),
            dynamic_rules_text=self._topic_prompt_sections_builder.build_rules_text(shortlist_ids),
            dynamic_examples_text=self._topic_prompt_sections_builder.build_examples_text(shortlist_ids),
            history_text=history_text,
            user_query=user_query,
            session_state_json=json.dumps(session_state.as_dict(), ensure_ascii=False),
            topic_title_map_json=json.dumps(self._topic_catalog.title_map(shortlist_ids), ensure_ascii=False),
            context_understanding_text=self._prompt_manager.build_context_understanding_text(
                gist=context_result.gist,
                meaning=context_result.meaning,
            )
            + (
                f"\n- Agent Zero turn_type: {getattr(context_result, 'turn_type', 'other')}"
                f"\n- Agent Zero turn_subtype: {getattr(context_result, 'turn_subtype', '') or '-'}"
                f"\n- Agent Zero semantic_flags: {', '.join(getattr(context_result, 'semantic_flags', []) or []) or '-'}"
            )
            + followup_prompt_text,
        )
        dialog_text = self._prompt_manager.build_dialog_text(history_text=history_text, user_query=user_query)

        result = self._topic_classifier.classify(
            prompt,
            dialog_text=dialog_text,
            user_query=user_query,
            context_signals=context_signals,
            session_state=session_state,
        )
        result = self._normalize_partial_epc_topic(
            result=result,
            user_query=user_query,
            slot_trace=slot_trace,
        )
        compatibility = self._semantic_compatibility_validator.validate(
            semantic_frame=getattr(context_result, "semantic_frame", None),
            topic_ids=list(result.topic_ids),
            shortlist_ids=shortlist_ids,
            state=session_state,
        )
        result = self._apply_semantic_compatibility(
            result=result,
            compatibility=compatibility,
            session_state=session_state,
        )
        result = self._apply_context_understanding_state(
            result=result,
            session_state=session_state,
            context_result=context_result,
        )
        result.diagnostics["user_query"] = user_query
        result.diagnostics["agent_zero"] = {
            "gist": context_result.gist,
            "meaning": context_result.meaning,
            "turn_type": getattr(context_result, "turn_type", "other"),
            "turn_subtype": getattr(context_result, "turn_subtype", ""),
            "confidence": getattr(context_result, "confidence", 0.0),
            "semantic_flags": list(getattr(context_result, "semantic_flags", []) or []),
            "semantic_frame": dict(semantic_frame),
            "fallback_used": context_result.fallback_used,
            "fallback_reason": context_result.fallback_reason,
        }
        result.diagnostics["agent_zero_trace"] = {
            "gist": context_result.gist,
            "meaning": context_result.meaning,
            "turn_type": getattr(context_result, "turn_type", "other"),
            "turn_subtype": getattr(context_result, "turn_subtype", ""),
            "confidence": getattr(context_result, "confidence", 0.0),
            "semantic_flags": list(getattr(context_result, "semantic_flags", []) or []),
            "semantic_frame": dict(semantic_frame),
            "fallback_used": context_result.fallback_used,
            "fallback_reason": context_result.fallback_reason,
            "raw_response": context_result.raw_response,
            "parsed_json": context_result.parsed_json,
            "validation_error": context_result.validation_error,
            "schema_retry_used": context_result.schema_retry_used,
        }
        result.diagnostics["context_understanding"] = {
            "gist": context_result.gist,
            "meaning": context_result.meaning,
            "semantic_frame": dict(semantic_frame),
            "raw_response": context_result.raw_response,
            "parsed_json": context_result.parsed_json,
            "fallback_used": context_result.fallback_used,
            "fallback_reason": context_result.fallback_reason,
            "json_extracted_from_wrapped_response": context_result.json_extracted_from_wrapped_response,
            "schema_retry_used": context_result.schema_retry_used,
            "validation_error": context_result.validation_error,
        }
        result.diagnostics["semantic_frame"] = dict(semantic_frame)
        result.diagnostics["shortlist"] = [item.as_dict() for item in shortlist]
        result.diagnostics["shortlist_topic_ids"] = shortlist_ids
        result.diagnostics["shortlist_trace"] = {
            "full_scores": self._topic_shortlist_builder.get_last_full_shortlist_scores(),
            "selected_scores": self._topic_shortlist_builder.get_last_selected_shortlist_scores(),
            "routing_trace": self._topic_shortlist_builder.get_last_semantic_routing_trace(),
        }
        result.diagnostics["semantic_compatibility_trace"] = compatibility.as_dict()
        result.diagnostics["slot_extraction_trace"] = dict(slot_trace)
        turn_analysis = result.diagnostics.setdefault("turn_analysis", {})
        if isinstance(turn_analysis, dict):
            slots = turn_analysis.setdefault("slots", {})
            if isinstance(slots, dict):
                slots.update(slot_trace.get("slots", {}) if isinstance(slot_trace.get("slots", {}), dict) else {})
        result.diagnostics["followup_trace"] = (
            dict(followup_resolution.trace) if followup_resolution is not None else {}
        )
        result.diagnostics["conversation_reasoning_trace"] = {
            "agent_zero": dict(result.diagnostics.get("agent_zero_trace", {})),
            "semantic_frame": dict(semantic_frame),
            "semantic_compatibility": compatibility.as_dict(),
            "shortlist": dict(result.diagnostics.get("shortlist_trace", {})),
            "slot_extraction": dict(slot_trace),
            "followup": dict(result.diagnostics.get("followup_trace", {})),
        }
        result.diagnostics["context_signal_trace"] = (
            self._context_signal_extractor.build_trace(context_signals)
            if self._context_signal_extractor is not None
            else {}
        )
        result.diagnostics["final_prompt"] = prompt
        result.diagnostics.setdefault("parsed_result", result.diagnostics.get("parsed_json", {}))

        first_agent_output = result.diagnostics.get("first_agent_output")
        if isinstance(first_agent_output, dict):
            first_agent_output["context_understanding"] = {
                "gist": context_result.gist,
                "meaning": context_result.meaning,
                "fallback_used": context_result.fallback_used,
                "fallback_reason": context_result.fallback_reason,
            }
        first_agent_trace = result.diagnostics.get("first_agent_trace")
        if isinstance(first_agent_trace, dict):
            pipeline_steps = first_agent_trace.get("pipeline_steps")
            if isinstance(pipeline_steps, list):
                pipeline_steps.insert(
                    0,
                    {
                        "step": "context_understanding",
                        "actor": "ContextUnderstandingAgent",
                        "target": "first_agent_output.context_understanding",
                        "status": "fallback" if context_result.fallback_used else "ok",
                        "before": {"dialog_text_present": bool(str(history_text).strip()), "user_query": user_query},
                        "after": {"gist": context_result.gist, "meaning": context_result.meaning},
                    },
                )
        state_trace = result.diagnostics.get("state_trace")
        if isinstance(state_trace, list):
            state_trace.insert(
                0,
                {
                    "step": "context_understanding",
                    "status": "fallback" if context_result.fallback_used else "ok",
                    "gist": context_result.gist,
                    "meaning": context_result.meaning,
                    "fallback_used": context_result.fallback_used,
                    "fallback_reason": context_result.fallback_reason,
                },
            )
        return prompt, result

    def analyze_context(self, history_text: str, user_query: str):
        return self._understand_context(history_text=history_text, user_query=user_query)

    def build_from_dialog_act(
        self,
        *,
        history_text: str,
        user_query: str,
        decision,
        context_result,
        slot_trace: dict[str, object],
        product_context,
        session_state: SessionState,
    ) -> tuple[str, TopicClassificationResult]:
        semantic_frame = (
            context_result.semantic_frame.as_dict()
            if getattr(context_result, "semantic_frame", None) is not None
            else {}
        )
        context_signals, followup_resolution, _ = self._resolve_followup(
            history_text=history_text,
            user_query=user_query,
            session_state=session_state,
            slot_trace=slot_trace,
            context_result=context_result,
        )
        snapshot = session_state.as_dict()
        snapshot.update(dict(getattr(decision, "extra_state_patch", {}) or {}))
        diagnostics = {
            "user_query": user_query,
            "classifier_source": "dialog_act_router",
            "dialog_act_trace": dict(getattr(decision, "trace", {}) or {}),
            "agent_zero": {
                "gist": context_result.gist,
                "meaning": context_result.meaning,
                "turn_type": getattr(context_result, "turn_type", "other"),
                "turn_subtype": getattr(context_result, "turn_subtype", ""),
                "confidence": getattr(context_result, "confidence", 0.0),
                "semantic_flags": list(getattr(context_result, "semantic_flags", []) or []),
                "semantic_frame": dict(semantic_frame),
                "fallback_used": context_result.fallback_used,
                "fallback_reason": context_result.fallback_reason,
            },
            "agent_zero_trace": {
                "gist": context_result.gist,
                "meaning": context_result.meaning,
                "turn_type": getattr(context_result, "turn_type", "other"),
                "turn_subtype": getattr(context_result, "turn_subtype", ""),
                "confidence": getattr(context_result, "confidence", 0.0),
                "semantic_flags": list(getattr(context_result, "semantic_flags", []) or []),
                "semantic_frame": dict(semantic_frame),
                "fallback_used": context_result.fallback_used,
                "fallback_reason": context_result.fallback_reason,
                "raw_response": context_result.raw_response,
                "parsed_json": context_result.parsed_json,
                "validation_error": context_result.validation_error,
                "schema_retry_used": context_result.schema_retry_used,
            },
            "context_understanding": {
                "gist": context_result.gist,
                "meaning": context_result.meaning,
                "semantic_frame": dict(semantic_frame),
                "raw_response": context_result.raw_response,
                "parsed_json": context_result.parsed_json,
                "fallback_used": context_result.fallback_used,
                "fallback_reason": context_result.fallback_reason,
                "json_extracted_from_wrapped_response": context_result.json_extracted_from_wrapped_response,
                "schema_retry_used": context_result.schema_retry_used,
                "validation_error": context_result.validation_error,
            },
            "semantic_frame": dict(semantic_frame),
            "slot_extraction_trace": dict(slot_trace),
            "product_context_trace": product_context.as_dict(),
            "shortlist_trace": {
                "skipped": True,
                "reason": getattr(decision, "reason", "dialog_act_router_applied"),
            },
            "followup_trace": dict(followup_resolution.trace) if followup_resolution is not None else {},
            "turn_analysis": {
                "current_focus": decision.topic_ids[0] if decision.topic_ids else "unknown",
                "slots": dict(slot_trace.get("slots", {})) if isinstance(slot_trace.get("slots", {}), dict) else {},
            },
            "state_before": session_state.as_dict(),
            "state_after": snapshot,
            "state_diff": dict_diff(session_state.as_dict(), snapshot),
            "first_agent_output": {
                "topic_ids": list(decision.topic_ids),
                "reason": getattr(decision, "reason", "dialog_act_router_applied"),
                "state_snapshot": snapshot,
            },
            "first_agent_data": {
                "brands": {
                    "normalized": list(getattr(product_context, "mentioned_brands", []) or []),
                }
            },
            "conversation_reasoning_trace": {
                "agent_zero": {
                    "turn_type": getattr(context_result, "turn_type", "other"),
                    "turn_subtype": getattr(context_result, "turn_subtype", ""),
                },
                "semantic_frame": dict(semantic_frame),
                "semantic_compatibility": {
                    "skipped": True,
                    "reason": "dialog_act_router_applied",
                },
                "slot_extraction": dict(slot_trace),
                "followup": dict(followup_resolution.trace) if followup_resolution is not None else {},
                "dialog_act": dict(getattr(decision, "trace", {}) or {}),
            },
            "context_signal_trace": (
                self._context_signal_extractor.build_trace(context_signals)
                if self._context_signal_extractor is not None
                else {}
            ),
            "semantic_compatibility_trace": {
                "is_compatible": True,
                "severity": "NONE",
                "selected_topic": decision.topic_ids[0] if decision.topic_ids else "",
                "final_topic": decision.topic_ids[0] if decision.topic_ids else "",
                "allowed_topics": [],
                "override_applied": False,
                "fallback_topic_used": False,
                "reason": "DialogActRouter path bypasses classifier compatibility validation.",
                "skipped": True,
                "skip_reason": "dialog_act_router_applied",
            },
        }
        result = TopicClassificationResult(
            topic_ids=list(decision.topic_ids),
            confidence=max(float(getattr(context_result, "confidence", 0.0) or 0.0), 0.9),
            reason=getattr(decision, "reason", "dialog_act_router_applied"),
            planned_action=getattr(decision, "action_name", ""),
            state_snapshot=snapshot,
            diagnostics=diagnostics,
            classifier_source=getattr(decision, "classifier_source", "dialog_act_router"),
        )
        result = self._apply_context_understanding_state(
            result=result,
            session_state=session_state,
            context_result=context_result,
        )
        return "", result

    def _resolve_followup(
        self,
        *,
        history_text: str,
        user_query: str,
        session_state: SessionState,
        slot_trace: dict[str, object],
        context_result,
    ):
        context_signals = self._build_context_signals(
            user_query=user_query,
            history_text=history_text,
            session_state=session_state,
            context_result=context_result,
        )
        followup_resolution = (
            self._followup_resolver.resolve(
                user_query=user_query,
                session_state=session_state,
                context_signals=context_signals,
                slot_trace=slot_trace,
                agent_zero_turn_type=str(getattr(context_result, "turn_type", "") or ""),
            )
            if self._followup_resolver is not None
            else None
        )
        if followup_resolution is not None:
            context_signals.semantic_boost_topics.update(followup_resolution.suggested_topics)
        return context_signals, followup_resolution, self._build_followup_prompt_text(followup_resolution)

    def _build_context_signals(
        self,
        user_query: str,
        history_text: str,
        session_state: SessionState,
        context_result,
    ) -> ContextSignals:
        fallback = ContextSignals(
            user_query=user_query,
            meaning=context_result.meaning,
            gist=context_result.gist,
            semantic_flags=set(str(item).strip() for item in getattr(context_result, "semantic_flags", []) or [] if str(item).strip()),
            fallback_used=bool(context_result.fallback_used),
        )
        if self._context_signal_extractor is None:
            return fallback
        return self._context_signal_extractor.extract(
            user_query=user_query,
            meaning=context_result.meaning,
            gist=context_result.gist,
            history_text=history_text,
            session_state=session_state,
            fallback_used=bool(context_result.fallback_used),
        )

    def _apply_context_understanding_state(
        self,
        result: TopicClassificationResult,
        session_state: SessionState,
        context_result,
    ) -> TopicClassificationResult:
        context_state_patch = {
            "last_context_gist": context_result.gist,
            "last_context_meaning": context_result.meaning,
            "last_context_fallback_used": bool(context_result.fallback_used),
            "last_context_fallback_reason": str(context_result.fallback_reason or ""),
            "conversation_mode": str(
                getattr(getattr(context_result, "semantic_frame", None), "conversation_mode", "unknown") or "unknown"
            ),
            "user_goal": str(
                getattr(getattr(context_result, "semantic_frame", None), "user_goal", "unknown") or "unknown"
            ),
            "last_semantic_frame": (
                context_result.semantic_frame.as_dict()
                if getattr(context_result, "semantic_frame", None) is not None
                else {}
            ),
        }
        updated_snapshot = dict(result.state_snapshot)
        updated_snapshot.update(context_state_patch)
        result.state_snapshot = updated_snapshot

        diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        state_before = diagnostics.get("state_before", session_state.as_dict())
        if not isinstance(state_before, dict):
            state_before = session_state.as_dict()
        diagnostics["state_after"] = updated_snapshot
        diagnostics["state_diff"] = dict_diff(state_before, updated_snapshot)
        active_pipeline = diagnostics.get("active_pipeline", [])
        if isinstance(active_pipeline, list) and "context_understanding" not in active_pipeline:
            diagnostics["active_pipeline"] = ["context_understanding", *active_pipeline]
        first_agent_output = diagnostics.get("first_agent_output")
        if isinstance(first_agent_output, dict):
            first_agent_output["state_snapshot"] = updated_snapshot
        return result

    @staticmethod
    def _apply_semantic_compatibility(
        *,
        result: TopicClassificationResult,
        compatibility,
        session_state: SessionState,
    ) -> TopicClassificationResult:
        if compatibility.is_compatible or not compatibility.final_topic:
            return result
        final_topic = str(compatibility.final_topic).strip()
        if not final_topic:
            return result
        result.topic_ids = [final_topic]
        snapshot = dict(result.state_snapshot) if isinstance(result.state_snapshot, dict) else session_state.as_dict()
        snapshot["last_primary_topic"] = final_topic
        snapshot["last_topic_ids"] = [final_topic]
        snapshot["last_secondary_topics"] = []
        snapshot["last_focus_topic"] = final_topic
        snapshot["active_request_kind"] = final_topic
        result.state_snapshot = snapshot

        diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        diagnostics["semantic_compatibility_override"] = {
            "applied": True,
            "severity": compatibility.severity,
            "selected_topic": compatibility.selected_topic,
            "final_topic": final_topic,
            "reason": compatibility.reason,
        }
        parsed_result = diagnostics.get("parsed_result", {})
        if isinstance(parsed_result, dict):
            parsed_result["intent_1"] = {
                "intent_id": final_topic,
                "score": result.confidence,
                "reason": compatibility.reason,
            }
            parsed_result["intent_2"] = None
        first_agent_output = diagnostics.get("first_agent_output", {})
        if isinstance(first_agent_output, dict):
            first_agent_output["topic_ids"] = [final_topic]
            first_agent_output["reason"] = compatibility.reason
            first_agent_output["state_snapshot"] = snapshot
        result.diagnostics = diagnostics
        result.reason = compatibility.reason
        return result

    @classmethod
    def _normalize_partial_epc_topic(
        cls,
        *,
        result: TopicClassificationResult,
        user_query: str,
        slot_trace: dict[str, object],
    ) -> TopicClassificationResult:
        normalized_query = re.sub(r"\s+", " ", str(user_query or "").lower().replace("ё", "е")).strip()
        if not any(pattern.search(normalized_query) for pattern in cls._PARTIAL_EPC_PATTERNS):
            return result
        extracted_slots = slot_trace.get("slots", {}) if isinstance(slot_trace.get("slots", {}), dict) else {}
        brands = extracted_slots.get("brands", [])
        has_brand = bool(brands) if isinstance(brands, list) else bool(str(extracted_slots.get("brand", "")).strip())
        if has_brand:
            return result
        result.topic_ids = ["partial_catalog_request"]
        snapshot = dict(result.state_snapshot) if isinstance(result.state_snapshot, dict) else {}
        snapshot["last_primary_topic"] = "partial_catalog_request"
        snapshot["last_topic_ids"] = ["partial_catalog_request"]
        snapshot["last_secondary_topics"] = []
        snapshot["last_focus_topic"] = "partial_catalog_request"
        snapshot["active_request_kind"] = "partial_catalog_request"
        result.state_snapshot = snapshot
        diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        diagnostics["post_classification_normalization"] = {
            "applied": True,
            "reason": "partial_epc_lexical_guard",
            "normalized_topic_ids": ["partial_catalog_request"],
        }
        first_agent_output = diagnostics.get("first_agent_output", {})
        if isinstance(first_agent_output, dict):
            first_agent_output["topic_ids"] = ["partial_catalog_request"]
            first_agent_output["reason"] = str(first_agent_output.get("reason", "")).strip() or "Normalized by partial EPC lexical guard."
            first_agent_output["state_snapshot"] = snapshot
        parsed_result = diagnostics.get("parsed_result", {})
        if isinstance(parsed_result, dict):
            parsed_result["intent_1"] = {
                "intent_id": "partial_catalog_request",
                "score": parsed_result.get("intent_1", {}).get("score", result.confidence) if isinstance(parsed_result.get("intent_1", {}), dict) else result.confidence,
                "reason": "Normalized by partial EPC lexical guard.",
            }
            parsed_result["intent_2"] = None
        result.diagnostics = diagnostics
        return result

    def _understand_context(self, history_text: str, user_query: str):
        if self._context_understanding_agent is None:
            return self._fallback_context_understanding(user_query)
        return self._context_understanding_agent.understand(
            dialog_text=self._prompt_manager.build_dialog_text(history_text=history_text, user_query=user_query),
            user_query=user_query,
        )

    @staticmethod
    def _fallback_context_understanding(user_query: str):
        from the_First_Agent.Agent_Zero.models import ContextUnderstandingResult, build_semantic_frame

        return ContextUnderstandingResult(
            gist="Не удалось надежно определить суть диалога.",
            meaning=f"Последняя реплика клиента: {user_query}",
            semantic_frame=build_semantic_frame(
                user_query=user_query,
                gist="Не удалось надежно определить суть диалога.",
                meaning=f"Последняя реплика клиента: {user_query}",
                turn_type="other",
                confidence=0.0,
            ),
            raw_response="",
            parsed_json={},
            fallback_used=True,
            fallback_reason="context_understanding_agent_not_configured",
        )

    @staticmethod
    def _build_followup_prompt_text(followup_resolution) -> str:
        if followup_resolution is None or not followup_resolution.is_followup:
            return ""
        inherited_brand = str(followup_resolution.inherited_brand or "-").strip() or "-"
        suggested = ", ".join(followup_resolution.suggested_topics) if followup_resolution.suggested_topics else "-"
        return (
            "\n- Follow-up контекст: да."
            f"\n- Тип follow-up: {followup_resolution.followup_type}"
            f"\n- Наследованный бренд: {inherited_brand}"
            f"\n- Suggested topics: {suggested}"
            f"\n- Причина: {followup_resolution.reason}"
        )
