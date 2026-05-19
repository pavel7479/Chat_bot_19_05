from __future__ import annotations

from json import JSONDecoder

from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from the_First_Agent.Agent_Zero.models import (
    ContextUnderstandingResult,
    ContextUnderstandingSchema,
    build_semantic_frame,
)


class ContextUnderstandingParser:
    _FALLBACK_GIST = "Не удалось надежно определить суть диалога."

    def __init__(self) -> None:
        self._json_parser = JsonOutputParser()

    def parse(
        self,
        raw_response: str,
        user_query: str,
        *,
        schema_retry_used: bool = False,
    ) -> ContextUnderstandingResult:
        raw = str(raw_response or "").strip()
        if not raw:
            return self.build_fallback(
                user_query=user_query,
                raw_response=str(raw_response or ""),
                parsed_json={},
                fallback_reason="empty_response",
                schema_retry_used=schema_retry_used,
                validation_error="empty_response",
            )

        extracted_json, extracted_from_wrapped = self._extract_json_object(raw)
        if extracted_json is None:
            return self.build_fallback(
                user_query=user_query,
                raw_response=raw,
                parsed_json={},
                fallback_reason="json_object_not_found",
                schema_retry_used=schema_retry_used,
                validation_error="json_object_not_found",
            )

        try:
            parsed = self._json_parser.parse(extracted_json)
        except Exception as error:
            fallback_reason = self._parse_error_reason(extracted_json)
            return self.build_fallback(
                user_query=user_query,
                raw_response=raw,
                parsed_json={},
                fallback_reason=fallback_reason,
                json_extracted_from_wrapped_response=extracted_from_wrapped,
                schema_retry_used=schema_retry_used,
                validation_error=f"{fallback_reason}:{error.__class__.__name__}",
            )

        if not isinstance(parsed, dict):
            return self.build_fallback(
                user_query=user_query,
                raw_response=raw,
                parsed_json={},
                fallback_reason="json_root_is_not_object",
                json_extracted_from_wrapped_response=extracted_from_wrapped,
                schema_retry_used=schema_retry_used,
                validation_error="json_root_is_not_object",
            )

        try:
            schema = ContextUnderstandingSchema.model_validate(parsed)
        except ValidationError as error:
            fallback_reason = self._validation_fallback_reason(error)
            return self.build_fallback(
                user_query=user_query,
                raw_response=raw,
                parsed_json=dict(parsed),
                fallback_reason=fallback_reason,
                json_extracted_from_wrapped_response=extracted_from_wrapped,
                schema_retry_used=schema_retry_used,
                validation_error=f"schema_validation_failed:{error.errors()}",
            )

        gist = schema.gist.strip()
        meaning = schema.meaning.strip()
        if not gist or not meaning:
            return self.build_fallback(
                user_query=user_query,
                raw_response=raw,
                parsed_json=dict(parsed),
                fallback_reason="empty_required_field",
                json_extracted_from_wrapped_response=extracted_from_wrapped,
                schema_retry_used=schema_retry_used,
                validation_error="empty_required_field:gist_or_meaning",
            )

        return ContextUnderstandingResult(
            gist=gist,
            meaning=meaning,
            turn_type=str(schema.turn_type or "other").strip() or "other",
            turn_subtype=str(schema.turn_subtype or "").strip(),
            confidence=float(schema.confidence or 0.0),
            semantic_flags=[str(item).strip() for item in schema.semantic_flags if str(item).strip()],
            semantic_frame=build_semantic_frame(
                user_query=user_query,
                gist=gist,
                meaning=meaning,
                turn_type=str(schema.turn_type or "other").strip() or "other",
                confidence=float(schema.confidence or 0.0),
                provided_frame=schema.semantic_frame,
            ),
            raw_response=raw,
            parsed_json=dict(parsed),
            fallback_used=False,
            fallback_reason="",
            json_extracted_from_wrapped_response=extracted_from_wrapped,
            schema_retry_used=schema_retry_used,
            validation_error="",
        )

    @classmethod
    def build_fallback(
        cls,
        user_query: str,
        raw_response: str,
        parsed_json: dict[str, object] | None,
        fallback_reason: str,
        json_extracted_from_wrapped_response: bool = False,
        schema_retry_used: bool = False,
        validation_error: str = "",
        gist: str = "",
        meaning: str = "",
    ) -> ContextUnderstandingResult:
        return ContextUnderstandingResult(
            gist=str(gist or "").strip() or cls._FALLBACK_GIST,
            meaning=str(meaning or "").strip() or f"Последняя реплика клиента: {user_query}",
            turn_type="other",
            turn_subtype="",
            confidence=0.0,
            semantic_flags=[],
            semantic_frame=build_semantic_frame(
                user_query=user_query,
                gist=str(gist or "").strip() or cls._FALLBACK_GIST,
                meaning=str(meaning or "").strip() or f"Последняя реплика клиента: {user_query}",
                turn_type="other",
                confidence=0.0,
            ),
            raw_response=str(raw_response or ""),
            parsed_json=dict(parsed_json or {}),
            fallback_used=True,
            fallback_reason=fallback_reason,
            json_extracted_from_wrapped_response=json_extracted_from_wrapped_response,
            schema_retry_used=schema_retry_used,
            validation_error=validation_error,
        )

    @staticmethod
    def _extract_json_object(text: str) -> tuple[str | None, bool]:
        decoder = JSONDecoder()
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                _, end = decoder.raw_decode(text[index:])
            except Exception:
                continue
            extracted = text[index : index + end]
            wrapped = index != 0 or end != len(text[index:])
            return extracted, wrapped
        return None, False

    @staticmethod
    def _parse_error_reason(text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return "empty_response"
        if "{" not in stripped and "[" not in stripped:
            return "json_object_not_found"
        if stripped.startswith("["):
            return "json_root_is_not_object"
        return "json_decode_error"

    @staticmethod
    def _validation_fallback_reason(error: ValidationError) -> str:
        errors = error.errors()
        if not errors:
            return "schema_validation_failed"
        allowed_fields = {
            "gist",
            "meaning",
            "turn_type",
            "turn_subtype",
            "confidence",
            "semantic_flags",
            "semantic_frame",
        }
        empty_field_errors = {
            "string_too_short",
            "too_short",
        }
        if all(
            tuple(item.get("loc", ())) in {("gist",), ("meaning",)}
            and str(item.get("type", "")) in empty_field_errors
            for item in errors
        ):
            return "empty_required_field"
        if all(tuple(item.get("loc", ())) and tuple(item.get("loc", ()))[0] in allowed_fields for item in errors):
            return "schema_validation_failed"
        return "schema_validation_failed"
