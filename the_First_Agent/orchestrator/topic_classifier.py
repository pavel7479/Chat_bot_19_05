from __future__ import annotations

import re
from pathlib import Path

from src.core.diff_utils import dict_diff
from src.core.interfaces import LLMProvider
from src.core.models import ContextSignals, SessionState, TopicClassificationResult
from the_First_Agent.config.intent_config_loader import IntentConfigLoader
from the_First_Agent.parsing.direct_parser import DirectClassificationParser
from the_First_Agent.preprocessing.input_normalizer import InputNormalizationService


class TopicClassifier:
    _SCHEMA_RETRY_LIMIT = 1
    _SCHEMA_RETRY_PREFIX = (
        "\n\nТвоя предыдущая попытка нарушила формат JSON.\n"
        "Ошибка схемы:\n"
        "{validation_error}\n\n"
        "Исправь ответ и верни только JSON точно в этом формате:\n"
        "- intent_1 обязателен\n"
        "- intent_2 либо null, либо объект\n"
        "- внутри intent можно использовать только поля intent_id, score, reason\n"
        "- не используй поля label, intent, confidence\n"
        "- не добавляй текст вне JSON\n"
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        topic_ids: set[str],
        topic_titles_by_id: dict[str, str] | None = None,
        intents_config_path: Path | None = None,
        brands_file_path: Path | None = None,
    ) -> None:
        self._llm = llm_provider
        self._topic_ids = topic_ids
        self._topic_titles_by_id = dict(topic_titles_by_id or {})
        project_root = Path(__file__).resolve().parents[2]
        intents_path = intents_config_path or (project_root / "src/config/config_intents.yaml")
        brands_path = brands_file_path or (project_root / "src/config/brands.yaml")
        self._intent_config = IntentConfigLoader(intents_path, brands_path).load()
        self._brand_aliases = [
            str(item)
            for item in self._intent_config.get("signals", {}).get("brand_aliases", [])
            if str(item).strip()
        ]
        self._normalizer = InputNormalizationService()
        self._parser = DirectClassificationParser(self._topic_titles_by_id)

    def classify(
        self,
        prompt: str,
        *,
        dialog_text: str,
        user_query: str,
        context_signals: ContextSignals | None = None,
        session_state: SessionState | None = None,
    ) -> TopicClassificationResult:
        normalized = self._normalizer.normalize(user_query)
        normalized_query = str(normalized["normalized_query"])
        parsed_result, llm_attempts = self._generate_with_schema_retry(prompt)
        topic_ids = list(parsed_result["topic_ids"]) or [self._fallback_topic_id(str(normalized["lowered_query"]))]
        parser_fallback_reason = str(parsed_result.get("fallback_reason", "")).strip()
        resolved_brand, brand_resolution_trace = self._resolve_last_brand(user_query, session_state)
        state_after = self._build_state_snapshot(
            session_state=session_state,
            topic_ids=topic_ids,
            user_query=user_query,
            resolved_brand=resolved_brand,
        )
        active_pipeline = [
            "receive_context",
            "normalize_input",
            "llm_generate_json",
            "parse_model_json",
            "build_state_snapshot",
        ]
        state_before = session_state.as_dict() if session_state else SessionState().as_dict()
        state_after_dict = state_after.as_dict()
        confidence = float(parsed_result["confidence"])
        reason = str(parsed_result["reason"])
        intent_scores = list(parsed_result.get("intent_scores", []))
        intent_details = list(parsed_result.get("intent_details", []))
        intent_reasons = list(parsed_result.get("intent_reasons", []))
        schema_retry_used = len(llm_attempts) > 1
        if not intent_scores:
            intent_scores = [{"intent": topic_id, "score": confidence} for topic_id in topic_ids]

        first_agent_output = {
            "original_query": user_query,
            "normalized_query": normalized_query,
            "prompt": prompt,
            "raw_llm_response": parsed_result["raw"],
            "parsed_json": parsed_result["parsed"],
            "topic_ids": list(topic_ids),
            "intent_scores": intent_scores,
            "intent_details": intent_details,
            "intent_reasons": intent_reasons,
            "confidence": confidence,
            "reason": reason,
            "state_snapshot": state_after_dict,
            "llm_attempts": llm_attempts,
            "schema_retry_used": schema_retry_used,
        }
        first_agent_trace = {
            "pipeline_steps": [
                {
                    "step": "receive_context",
                    "actor": "TopicClassifier",
                    "target": "first_agent_output",
                    "status": "ok",
                    "before": {},
                    "after": {"user_query": user_query, "dialog_text_present": bool(str(dialog_text).strip())},
                },
                {
                    "step": "normalize_input",
                    "actor": "InputNormalizationService",
                    "target": "first_agent_output",
                    "status": "ok",
                    "before": {"query": user_query},
                    "after": {"normalized_query": normalized_query},
                },
                {
                    "step": "llm_generate_json",
                    "actor": "LLMProvider",
                    "target": "first_agent_output",
                    "status": "ok",
                    "before": {},
                    "after": {
                        "raw_response_present": bool(str(parsed_result["raw"]).strip()),
                        "attempt_count": len(llm_attempts),
                        "schema_retry_used": schema_retry_used,
                    },
                },
                {
                    "step": "parse_model_json",
                    "actor": "DirectClassificationParser",
                    "target": "first_agent_output",
                    "status": "ok",
                    "before": {"raw_llm_response": parsed_result["raw"]},
                    "after": {
                        "topic_ids": list(topic_ids),
                        "confidence": confidence,
                        "fallback_used": bool(parsed_result["fallback_used"]),
                        "fallback_reason": parser_fallback_reason,
                        "validation_errors": list(parsed_result.get("validation_errors", [])),
                        "schema_retry_used": schema_retry_used,
                    },
                },
                {
                    "step": "build_state_snapshot",
                    "actor": "TopicClassifier",
                    "target": "state_snapshot",
                    "status": "ok",
                    "before": state_before,
                    "after": state_after_dict,
                    "diff": dict_diff(state_before, state_after_dict),
                },
            ],
        }

        diagnostics = {
            "active_pipeline": active_pipeline,
            "pipeline_version": "first_agent_direct_v2_simple",
            "pipeline_contract": {
                "source_of_truth": "model_json",
                "post_model_topic_adjustments_enabled": False,
                "rewrite_enabled": False,
                "input_normalization_enabled": True,
            },
            "prompt_context_source": "direct_input",
            "context_signals": (
                context_signals.as_dict()
                if context_signals is not None
                else ContextSignals(user_query=user_query, gist="", meaning="").as_dict()
            ),
            "final_prompt": prompt,
            "raw_llm_response": parsed_result["raw"],
            "parsed_result": parsed_result["parsed"],
            "parsed_json": parsed_result["parsed"],
            "normalization_trace": normalized,
            "brand_resolution_trace": brand_resolution_trace,
            "first_agent_output": first_agent_output,
            "first_agent_data": {},
            "first_agent_trace": first_agent_trace,
            "validation_errors": list(parsed_result.get("validation_errors", [])),
            "parser_fallback_reason": parser_fallback_reason,
            "llm_attempts": llm_attempts,
            "schema_retry_used": schema_retry_used,
            "state_before": state_before,
            "state_after": state_after_dict,
            "state_diff": dict_diff(state_before, state_after_dict),
            "state_trace": [
                {"step": "receive_context", "user_query": user_query, "dialog_text_present": bool(str(dialog_text).strip())},
                {"step": "normalize_input", "normalized_query": normalized_query},
                {
                    "step": "llm_generate_json",
                    "raw_response_present": bool(str(parsed_result["raw"]).strip()),
                    "attempt_count": len(llm_attempts),
                    "schema_retry_used": schema_retry_used,
                },
                {
                    "step": "parse_model_json",
                    "topics": topic_ids,
                    "fallback_used": bool(parsed_result["fallback_used"]),
                    "fallback_reason": parser_fallback_reason,
                    "validation_errors": list(parsed_result.get("validation_errors", [])),
                },
                {"step": "build_state_snapshot", "last_primary_topic": topic_ids[0] if topic_ids else "out_of_scope_request"},
            ],
        }

        return TopicClassificationResult(
            topic_ids=topic_ids,
            confidence=confidence,
            reason=reason,
            rule_trace=[],
            state_snapshot=state_after_dict,
            diagnostics=diagnostics,
            classifier_source="direct_json",
            fallback_reason=parser_fallback_reason if parsed_result["fallback_used"] else "",
        )

    def _build_state_snapshot(
        self,
        session_state: SessionState | None,
        topic_ids: list[str],
        user_query: str,
        resolved_brand: str,
    ) -> SessionState:
        base = session_state.as_dict() if session_state else SessionState().as_dict()
        primary_topic = topic_ids[0] if topic_ids else "out_of_scope_request"
        secondary_topics = list(topic_ids[1:]) if len(topic_ids) > 1 else []
        base["last_primary_topic"] = primary_topic
        base["last_topic_ids"] = list(topic_ids)
        base["last_secondary_topics"] = secondary_topics
        base["last_focus_topic"] = primary_topic
        base["last_bot_question_type"] = base.get("last_question_type", "unknown")
        base["active_request_kind"] = "multi_intent" if len(topic_ids) > 1 else primary_topic
        base["last_mentioned_brand"] = resolved_brand or self._find_last_brand_in_text(user_query)
        return SessionState(**base)

    def _resolve_last_brand(
        self,
        user_query: str,
        session_state: SessionState | None,
    ) -> tuple[str, dict[str, str]]:
        query_brand = self._find_last_brand_in_text(user_query)
        if query_brand:
            return query_brand, {"source": "query", "brand": query_brand}
        session_brand = str(session_state.last_mentioned_brand).strip() if session_state else ""
        if session_brand:
            return session_brand, {"source": "session", "brand": session_brand}
        return "", {"source": "none", "brand": ""}

    def _find_last_brand_in_text(self, text: str) -> str:
        haystack = str(text or "").lower()
        last_match = ""
        for alias in self._brand_aliases:
            if re.search(rf"\b{re.escape(alias)}\b", haystack):
                last_match = alias
        return last_match.title() if last_match else ""

    def _generate_with_schema_retry(self, prompt: str) -> tuple[dict[str, object], list[dict[str, object]]]:
        attempts: list[dict[str, object]] = []
        effective_prompt = prompt
        parsed_result: dict[str, object] = {
            "raw": "",
            "parsed": {},
            "topic_ids": [],
            "intent_scores": [],
            "intent_reasons": [],
            "intent_details": [],
            "confidence": 0.0,
            "reason": "Fallback: invalid classifier output.",
            "fallback_used": True,
            "fallback_reason": "json_parse_failed",
            "validation_errors": [],
        }
        for attempt_index in range(self._SCHEMA_RETRY_LIMIT + 1):
            raw_response = self._llm.generate_json(effective_prompt)
            parsed_result = self._parser.parse(raw_response)
            validation_errors = list(parsed_result.get("validation_errors", []))
            fallback_reason = str(parsed_result.get("fallback_reason", "")).strip()
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "feedback_applied": attempt_index > 0,
                    "prompt_suffix": "schema_retry" if attempt_index > 0 else "",
                    "raw_llm_response": raw_response,
                    "parsed_json": parsed_result.get("parsed", {}),
                    "validation_errors": validation_errors,
                    "fallback_reason": fallback_reason,
                }
            )
            if not validation_errors or attempt_index >= self._SCHEMA_RETRY_LIMIT:
                return parsed_result, attempts
            effective_prompt = self._build_schema_feedback_prompt(prompt, validation_errors)
        return parsed_result, attempts

    @classmethod
    def _build_schema_feedback_prompt(cls, prompt: str, validation_errors: list[object]) -> str:
        validation_error = str(validation_errors[0]).strip() if validation_errors else "Неизвестная ошибка схемы."
        return prompt + cls._SCHEMA_RETRY_PREFIX.format(validation_error=validation_error)

    @staticmethod
    def _fallback_topic_id(lowered_query: str) -> str:
        if not lowered_query or re.fullmatch(r"[\W_]+", lowered_query):
            return "nonsense_input"
        return "out_of_scope_request"
