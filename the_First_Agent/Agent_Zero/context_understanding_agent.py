from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

from src.core.interfaces import LLMProvider
from the_First_Agent.Agent_Zero.models import ContextUnderstandingResult
from the_First_Agent.Agent_Zero.parser import ContextUnderstandingParser


class ContextUnderstandingAgent:
    _SCHEMA_RETRY_LIMIT = 2
    _RETRYABLE_REASONS = {
        "empty_response",
        "json_object_not_found",
        "json_decode_error",
        "json_root_is_not_object",
        "schema_validation_failed",
        "empty_required_field",
    }
    _SCHEMA_RETRY_SUFFIX = (
        "\n\nТвоя предыдущая попытка вернула невалидный JSON.\n"
        "Верни только JSON с полями: gist, meaning, turn_type, turn_subtype, confidence, semantic_flags.\n"
        "Можно дополнительно вернуть semantic_frame, если уверен в смысловом режиме реплики.\n"
        "Не добавляй другие поля.\n"
        "Не добавляй текст вне JSON.\n"
        "gist: одно короткое предложение, не больше 25 слов.\n"
        "meaning: одно короткое предложение, не больше 25 слов.\n"
        "turn_type: одно значение из списка greeting, service_discovery_continue, pricing_request, brand_list_for_tis, pricing_followup, clarification_pushback, human_escalation, abuse_or_frustration, noise, other.\n"
        "turn_subtype: короткая строка или пустая строка.\n"
        "confidence: число от 0 до 1.\n"
        "semantic_flags: массив коротких строк.\n"
        "semantic_frame: объект с полями conversation_mode, user_goal, is_followup, is_topic_switch, language, confidence, gist, meaning.\n"
    )

    def __init__(
        self,
        llm: LLMProvider,
        prompt_path: Path,
        logger: Any | None = None,
    ) -> None:
        self._llm = llm
        self._prompt_path = Path(prompt_path)
        self._prompt_template = Template(
            self._prompt_path.read_text(encoding="utf-8")
            .replace("{dialog_text}", "$dialog_text")
            .replace("{user_query}", "$user_query")
        )
        self._parser = ContextUnderstandingParser()
        self._logger = logger

    def build_prompt(
        self,
        dialog_text: str,
        user_query: str,
    ) -> str:
        base_prompt = self._prompt_template.safe_substitute(
            dialog_text=str(dialog_text),
            user_query=str(user_query),
        )
        schema_block = (
            "\n\nВерни только JSON строго такого вида:\n"
            "{\n"
            '  "gist": "...",\n'
            '  "meaning": "...",\n'
            '  "turn_type": "greeting|service_discovery_continue|pricing_request|brand_list_for_tis|pricing_followup|clarification_pushback|human_escalation|abuse_or_frustration|noise|other",\n'
            '  "turn_subtype": "...",\n'
            '  "confidence": 0.0,\n'
            '  "semantic_flags": [],\n'
            '  "semantic_frame": {\n'
            '    "conversation_mode": "unknown|discovery|product_choice|pricing|purchase|support|renewal|manager|complaint|security|out_of_scope|smalltalk",\n'
            '    "user_goal": "greet|ask_product_list|ask_product_recommendation|ask_price|ask_purchase_steps|ask_support|ask_manager|ask_benefits|ask_limitations|complain_or_distrust|ask_free_or_bypass_payment|unknown",\n'
            '    "is_followup": false,\n'
            '    "is_topic_switch": false,\n'
            '    "language": "ru|en",\n'
            '    "confidence": 0.0,\n'
            '    "gist": "...",\n'
            '    "meaning": "..."\n'
            "  }\n"
            "}\n"
            "Никакого текста вне JSON.\n"
            "semantic_frame можно опустить, если не уверен.\n"
            "Если бот только поздоровался или предложил рассказать о сервисе, реплики "
            "\"ну попробуй\", \"рассказывай\", \"да\" означают service_discovery_continue, а не demo_access.\n"
        )
        return f"{base_prompt}{schema_block}"

    def understand(
        self,
        dialog_text: str,
        user_query: str,
    ) -> ContextUnderstandingResult:
        prompt = self.build_prompt(dialog_text=dialog_text, user_query=user_query)
        self._log("context_understanding_prompt_built", {"prompt": prompt, "user_query": user_query})
        llm_attempts: list[dict[str, object]] = []
        try:
            raw_response = str(self._llm.generate_json(prompt) or "")
        except Exception as error:
            result = ContextUnderstandingParser.build_fallback(
                user_query=user_query,
                raw_response="",
                parsed_json={},
                fallback_reason="context_understanding_llm_error",
                validation_error=str(error),
            )
            self._log(
                "context_understanding_fallback",
                {
                    "error": str(error),
                    "fallback_reason": result.fallback_reason,
                    "gist": result.gist,
                    "meaning": result.meaning,
                    "turn_type": result.turn_type,
                    "validation_error": result.validation_error,
                },
            )
            return result

        result = self._parser.parse(raw_response=raw_response, user_query=user_query)
        llm_attempts.append(
            {
                "attempt": 1,
                "feedback_applied": False,
                "raw_response": raw_response,
                "fallback_used": result.fallback_used,
                "fallback_reason": result.fallback_reason,
                "json_extracted_from_wrapped_response": result.json_extracted_from_wrapped_response,
                "validation_error": result.validation_error,
            }
        )
        if result.fallback_used and result.fallback_reason in self._RETRYABLE_REASONS:
            result = self._retry_on_schema_error(
                prompt=prompt,
                user_query=user_query,
                llm_attempts=llm_attempts,
            )
        self._log(
            "context_understanding_result",
            {
                "raw_response": result.raw_response,
                "parsed_json": result.parsed_json,
                "gist": result.gist,
                "meaning": result.meaning,
                "turn_type": result.turn_type,
                "turn_subtype": result.turn_subtype,
                "confidence": result.confidence,
                "semantic_flags": list(result.semantic_flags),
                "semantic_frame": result.semantic_frame.as_dict() if result.semantic_frame is not None else {},
                "fallback_used": result.fallback_used,
                "fallback_reason": result.fallback_reason,
                "json_extracted_from_wrapped_response": result.json_extracted_from_wrapped_response,
                "schema_retry_used": result.schema_retry_used,
                "validation_error": result.validation_error,
                "llm_attempts": llm_attempts,
            },
        )
        return result

    def _retry_on_schema_error(
        self,
        prompt: str,
        user_query: str,
        llm_attempts: list[dict[str, object]],
    ) -> ContextUnderstandingResult:
        retry_prompt = f"{prompt}{self._SCHEMA_RETRY_SUFFIX}"
        last_result: ContextUnderstandingResult | None = None
        for attempt_index in range(self._SCHEMA_RETRY_LIMIT):
            try:
                raw_retry = str(self._llm.generate_json(retry_prompt) or "")
            except Exception as error:
                result = ContextUnderstandingParser.build_fallback(
                    user_query=user_query,
                    raw_response="",
                    parsed_json={},
                    fallback_reason="context_understanding_llm_error",
                    schema_retry_used=True,
                    validation_error=str(error),
                )
                llm_attempts.append(
                    {
                        "attempt": attempt_index + 2,
                        "feedback_applied": True,
                        "raw_response": "",
                        "fallback_used": True,
                        "fallback_reason": result.fallback_reason,
                        "json_extracted_from_wrapped_response": False,
                        "validation_error": result.validation_error,
                    }
                )
                return result

            parsed_retry = self._parser.parse(
                raw_response=raw_retry,
                user_query=user_query,
                schema_retry_used=True,
            )
            llm_attempts.append(
                {
                    "attempt": attempt_index + 2,
                    "feedback_applied": True,
                    "raw_response": raw_retry,
                    "fallback_used": parsed_retry.fallback_used,
                    "fallback_reason": parsed_retry.fallback_reason,
                    "json_extracted_from_wrapped_response": parsed_retry.json_extracted_from_wrapped_response,
                    "validation_error": parsed_retry.validation_error,
                }
            )
            if not parsed_retry.fallback_used:
                return parsed_retry
            last_result = parsed_retry

        if last_result is not None:
            return last_result

        return ContextUnderstandingParser.build_fallback(
            user_query=user_query,
            raw_response="",
            parsed_json={},
            fallback_reason="schema_validation_failed",
            schema_retry_used=True,
            validation_error="schema_retry_exhausted",
        )

    def _log(self, event: str, payload: dict[str, object]) -> None:
        if self._logger is None:
            return
        if hasattr(self._logger, "info"):
            try:
                self._logger.info(event, extra={"context_understanding": payload})
                return
            except Exception:
                return
