from __future__ import annotations

import json

from the_First_Agent.parsing.output_schema import FirstAgentOutputSchema


class DirectClassificationParser:
    _INTENT_ID_ALIASES = {
        "human_request": "human_operator_request",
        "human_operator": "human_operator_request",
        "competitor_intent": "competitor_choice",
        "legal_entity_purchase": "legal_entity_purchase_flow",
        "specific_brand": "specific_brand_check",
        "brand_list": "brand_list_request",
        "company_info": "company_services_info",
        "mac_os": "macos_support",
    }

    def __init__(self, topic_titles_by_id: dict[str, str]) -> None:
        self._topic_titles_by_id = {
            str(topic_id).strip(): str(title).strip()
            for topic_id, title in dict(topic_titles_by_id).items()
            if str(topic_id).strip()
        }
        self._topic_ids = set(self._topic_titles_by_id.keys())
        self._normalized_topic_ids = {
            self._normalize_intent_id(topic_id): topic_id
            for topic_id in self._topic_ids
        }
        self._normalized_aliases = {
            self._normalize_intent_id(alias): canonical
            for alias, canonical in self._INTENT_ID_ALIASES.items()
            if canonical in self._topic_ids
        }

    def parse(self, raw: str) -> dict[str, object]:
        parsed, validation_error = self._safe_parse(raw)
        if validation_error:
            topics, intent_scores, intent_reasons, intent_details, extraction_meta = [], [], [], [], {
                "seen_intent_object": False,
                "seen_intent_id": False,
                "unknown_intent_ids": [],
            }
        else:
            topics, intent_scores, intent_reasons, intent_details, extraction_meta = self._extract_topics(parsed)
        reason = self._build_combined_reason(intent_reasons)
        primary_score = float(intent_scores[0]["score"]) if intent_scores else 0.0
        fallback_reason = self._resolve_fallback_reason(
            parsed=parsed,
            topics=topics,
            validation_error=validation_error,
            extraction_meta=extraction_meta,
        )
        return {
            "raw": raw,
            "parsed": parsed,
            "topic_ids": topics,
            "intent_scores": intent_scores,
            "intent_reasons": intent_reasons,
            "intent_details": intent_details,
            "confidence": primary_score,
            "reason": reason,
            "fallback_used": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "validation_errors": [validation_error] if validation_error else [],
        }

    def _extract_topics(
        self,
        parsed: dict[str, object],
    ) -> tuple[list[str], list[dict[str, object]], list[str], list[dict[str, object]], dict[str, object]]:
        topics: list[str] = []
        scores: list[dict[str, object]] = []
        reasons: list[str] = []
        details: list[dict[str, object]] = []
        seen_intent_object = False
        seen_intent_id = False
        unknown_intent_ids: list[str] = []
        for key in ("intent_1", "intent_2"):
            value = parsed.get(key)
            if not isinstance(value, dict):
                continue
            seen_intent_object = True
            raw_intent_id = str(value.get("intent_id", "")).strip()
            if raw_intent_id:
                seen_intent_id = True
            topic_id = self._canonical_intent_id(raw_intent_id)
            if not topic_id:
                if raw_intent_id:
                    unknown_intent_ids.append(raw_intent_id)
                continue
            if topic_id in topics:
                continue
            score = self._safe_score(value.get("score"))
            reason = str(value.get("reason", "")).strip() or self._build_reason_from_intent(topic_id)
            topics.append(topic_id)
            scores.append({"intent": topic_id, "score": score, "intent_id": topic_id})
            reasons.append(reason)
            details.append(
                {
                    "intent": topic_id,
                    "intent_id": topic_id,
                    "score": score,
                    "reason": reason,
                }
            )
        return (
            topics[:2],
            scores[:2],
            reasons[:2],
            details[:2],
            {
                "seen_intent_object": seen_intent_object,
                "seen_intent_id": seen_intent_id,
                "unknown_intent_ids": unknown_intent_ids[:2],
            },
        )

    @staticmethod
    def _resolve_fallback_reason(
        parsed: dict[str, object],
        topics: list[str],
        validation_error: str,
        extraction_meta: dict[str, object],
    ) -> str:
        if validation_error.startswith("SCHEMA_VALIDATION:"):
            return "schema_validation_failed"
        if validation_error or not parsed:
            return "json_parse_failed"
        if topics:
            return ""
        unknown_intent_ids = extraction_meta.get("unknown_intent_ids", [])
        if isinstance(unknown_intent_ids, list) and unknown_intent_ids:
            return "unknown_intent_id"
        if bool(extraction_meta.get("seen_intent_object")) and not bool(extraction_meta.get("seen_intent_id")):
            return "missing_intent_id"
        return "empty_topics_after_parse"

    @staticmethod
    def _build_reason_from_intent(intent_id: str) -> str:
        if intent_id:
            return f"Клиентский запрос отнесен к теме `{intent_id}`."
        return "Fallback: invalid classifier output."

    def _canonical_intent_id(self, intent_id: str) -> str:
        normalized = self._normalize_intent_id(intent_id)
        if not normalized:
            return ""
        direct = self._normalized_topic_ids.get(normalized, "")
        if direct:
            return direct
        return self._normalized_aliases.get(normalized, "")

    @staticmethod
    def _normalize_intent_id(value: str) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _build_combined_reason(intent_reasons: list[str]) -> str:
        normalized = [str(item).strip() for item in intent_reasons if str(item).strip()]
        if not normalized:
            return "Fallback: invalid classifier output."
        if len(normalized) == 1:
            return normalized[0]
        return "; ".join(normalized[:2])

    def _safe_parse(self, raw: str) -> tuple[dict[str, object], str]:
        cleaned = self._extract_json(raw)
        if not cleaned:
            return {}, "Пустой или нераспознанный JSON в ответе модели."
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            return {}, str(exc)

        if not isinstance(parsed, dict):
            return {}, "SCHEMA_VALIDATION: Корневой JSON-объект должен быть объектом."

        try:
            validated = FirstAgentOutputSchema.model_validate(parsed)
            return validated.model_dump(), ""
        except Exception as exc:
            return parsed, f"SCHEMA_VALIDATION: {exc}"

    @staticmethod
    def _extract_json(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                _, end_index = decoder.raw_decode(text[index:])
                candidate = text[index:index + end_index].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    return candidate
            except json.JSONDecodeError:
                continue
        return ""

    @staticmethod
    def _safe_score(value: object) -> float:
        try:
            parsed = float(value)
            return max(0.0, min(1.0, parsed))
        except (TypeError, ValueError):
            return 0.0
