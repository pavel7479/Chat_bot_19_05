from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError

from src.core.interfaces import LLMProvider
from src.intents.intent_prompt_topic_selector import IntentPromptTopicSelector
from src.intents.query_rewriter_service import QueryRewriterService


@dataclass(slots=True)
class ModelIntentScore:
    intent: str
    score: float
    evidence: str


class ModelIntentCandidateSchema(BaseModel):
    intent: str = Field(description="Topic id from allowed list")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score in [0..1]")
    evidence: str = Field(default="", description="Short reason")


class ModelIntentResponseSchema(BaseModel):
    candidates: list[ModelIntentCandidateSchema] = Field(default_factory=list)


class ModelSemanticProvider:
    """Optional model-backed semantic scorer for intent candidates.

    Stage-1 integration: infrastructure only. Can be enabled in config file.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        intents: list[str],
        intent_labels_ru: dict[str, str] | None = None,
        intent_hints: dict[str, dict[str, object]] | None = None,
        enabled: bool = False,
        top_k: int = 3,
        max_evidence_tokens: int = 12,
        prompt_template_path: Path | None = None,
        selector_top_n: int = 12,
        selector_min_score: float = 0.0,
        rewrite_enabled: bool = True,
    ) -> None:
        self._llm = llm_provider
        self._intents = [item for item in intents if item]
        self._enabled = bool(enabled)
        self._top_k = max(1, int(top_k))
        self._max_evidence_tokens = max(4, int(max_evidence_tokens))
        self._intent_labels_ru = dict(intent_labels_ru or {})
        self._intent_hints = dict(intent_hints or {})
        self._prompt_template = self._load_prompt_template(prompt_template_path)
        self._selector = IntentPromptTopicSelector(
            intents=self._intents,
            intent_labels_ru=self._intent_labels_ru,
            intent_hints=self._intent_hints,
            top_n=selector_top_n,
            min_score=selector_min_score,
        )
        self._rewriter = QueryRewriterService()
        self._rewrite_enabled = bool(rewrite_enabled)
        self._last_prompt_trace: dict[str, object] = {}
        self._output_parser = JsonOutputParser(pydantic_object=ModelIntentResponseSchema)
        self._response_schema_instructions = self._build_response_schema_instructions()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def score_intents(
        self,
        query: str,
        context_summary: dict[str, object] | None = None,
        selected_topics: list[str] | None = None,
        trace_id: str = "",
    ) -> list[ModelIntentScore]:
        if not self._enabled or not query.strip() or not self._intents:
            self._last_prompt_trace = {}
            return []
        context_payload = context_summary or {}
        if self._rewrite_enabled:
            rewrite = self._rewriter.rewrite(query, context_payload)
        else:
            rewrite = QueryRewriterService().rewrite(query, {})
        selection = self._selector.select(query, context_payload)
        # Model-first contract: pass all configured intents to model.
        # Selector diagnostics are kept in trace for observability only.
        intents_catalog_text = self._build_selected_payload(self._intents)
        previous_intents = self._extract_previous_intents(context_payload)
        dialog_history = self._format_dialog_history(context_payload)
        if not intents_catalog_text.strip():
            self._last_prompt_trace = {
                "original_query": query,
                "rewritten_query": rewrite.rewritten_query,
                "rewrite_changed": rewrite.changed,
                "rewrite_reason": rewrite.reason,
                "selected_intents": list(self._intents),
                "selected_reasons": dict(selection.selected_reasons),
                "dropped_reasons": dict(selection.dropped_reasons),
                "dialog_history": dialog_history,
                "previous_intents": list(previous_intents),
                "trace_id": trace_id,
                "parse_status": "skipped_empty_intent_catalog",
                "fallback_reason": "empty_intent_catalog",
            }
            return []
        prompt = self._build_prompt(
            query=query,
            rewritten_query=rewrite.rewritten_query,
            intents_catalog_text=intents_catalog_text,
            dialog_history=dialog_history,
            previous_intents=previous_intents,
        )
        self._last_prompt_trace = {
            "original_query": query,
            "rewritten_query": rewrite.rewritten_query,
            "rewrite_changed": rewrite.changed,
            "rewrite_reason": rewrite.reason,
            "selected_intents": list(self._intents),
            "selected_reasons": dict(selection.selected_reasons),
            "dropped_reasons": dict(selection.dropped_reasons),
            "dialog_history": dialog_history,
            "previous_intents": list(previous_intents),
            "full_prompt": prompt,
            "trace_id": trace_id,
        }
        try:
            raw = self._llm.generate_json(prompt)
            self._last_prompt_trace["raw_response"] = raw
            parsed = self._parse_with_langchain(raw)
            candidates = self._extract_candidates(parsed)
            normalized = self._normalize_candidates(candidates)[: self._top_k]
            if normalized:
                self._last_prompt_trace["parse_status"] = "ok"
                self._last_prompt_trace["retry_used"] = False
                return normalized
            self._last_prompt_trace["parse_status"] = "empty_after_validation"
            self._last_prompt_trace["validation_error"] = "no_valid_candidates"
            retry_prompt = self._build_retry_prompt(prompt)
            raw_retry = self._llm.generate_json(retry_prompt)
            self._last_prompt_trace["raw_retry_response"] = raw_retry
            parsed_retry = self._parse_with_langchain(raw_retry)
            candidates_retry = self._extract_candidates(parsed_retry)
            normalized_retry = self._normalize_candidates(candidates_retry)[: self._top_k]
            self._last_prompt_trace["retry_used"] = True
            if normalized_retry:
                self._last_prompt_trace["parse_status"] = "ok_after_retry"
                return normalized_retry
            self._last_prompt_trace["parse_status"] = "failed_after_retry"
            self._last_prompt_trace["fallback_reason"] = "parse_invalid"
            return []
        except Exception as exc:
            self._last_prompt_trace["parse_status"] = "exception"
            self._last_prompt_trace["validation_error"] = str(exc)
            self._last_prompt_trace["fallback_reason"] = "parse_exception"
            return []

    def _build_prompt(
        self,
        query: str,
        rewritten_query: str,
        intents_catalog_text: str,
        dialog_history: str,
        previous_intents: list[str],
    ) -> str:
        previous_intents_json = json.dumps(previous_intents, ensure_ascii=False)
        template = self._prompt_template
        if template:
            return template.format(
                dialog_history=dialog_history,
                previous_intents_json=previous_intents_json,
                rewritten_query=rewritten_query,
                user_query=query,
                intents_catalog_text=intents_catalog_text,
                format_instructions=self._response_schema_instructions,
            )
        return (
            "Ты строгий классификатор намерений.\n"
            "Разрешено использовать ТОЛЬКО intent из списка.\n"
            "Сначала определи главный смысл вопроса пользователя, затем выбери intent.\n"
            "Если запрос непонятный, используй intent 'nonsense_input'.\n"
            "Если запрос не по профилю компании, используй intent 'out_of_scope_request'.\n"
            f"{self._response_schema_instructions}\n"
            f"Диалог: {dialog_history}\n"
            f"Предыдущие намерения: {previous_intents_json}\n"
            f"Переформулировка: {rewritten_query}\n"
            f"Список intent:\n{intents_catalog_text}\n"
            f"Запрос: {query}"
        )

    def get_last_prompt_trace(self) -> dict[str, object]:
        return dict(self._last_prompt_trace)

    def _build_selected_payload(self, selected_intents: list[str]) -> str:
        selected_set = set(selected_intents)
        cards: list[str] = []
        idx = 1
        for intent in self._intents:
            if intent not in selected_set:
                continue
            label = self._intent_labels_ru.get(intent, "").strip() or intent
            choose_when = str(self._intent_hints.get(intent, {}).get("choose_when", "")).strip() or "—"
            not_choose_when = str(self._intent_hints.get(intent, {}).get("not_choose_when", "")).strip() or "—"
            cards.append(
                (
                    f"{idx}) Код: {intent}\n"
                    f"Название: {label}\n"
                    f"Выбирать, когда: {choose_when}\n"
                    f"Не выбирать, когда: {not_choose_when}"
                )
            )
            idx += 1
        return "\n\n".join(cards)

    @staticmethod
    def _format_dialog_history(context_summary: dict[str, object]) -> str:
        history = context_summary.get("history_tail", [])
        if not isinstance(history, list) or not history:
            return "клиент: (история пуста)"
        lines: list[str] = []
        for idx, item in enumerate(history):
            text = str(item).strip()
            if not text:
                continue
            if ":" in text:
                _, rhs = text.split(":", 1)
                text = rhs.strip()
            role = "клиент" if idx % 2 == 0 else "менеджер"
            lines.append(f"{role}: {text}")
        return "\n".join(lines) if lines else "клиент: (история пуста)"

    @staticmethod
    def _extract_previous_intents(context_summary: dict[str, object]) -> list[str]:
        prev = str(context_summary.get("previous_primary_topic", "")).strip()
        if not prev:
            return []
        return [prev]

    @staticmethod
    def _load_prompt_template(path: Path | None) -> str:
        if path is None or not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _parse_with_langchain(self, raw: str) -> dict[str, Any]:
        try:
            parsed = self._output_parser.parse(raw)
        except Exception as exc:
            self._last_prompt_trace["parse_error_stage"] = "langchain_parser"
            self._last_prompt_trace["validation_error"] = str(exc)
            return {}
        if isinstance(parsed, dict):
            return parsed
        self._last_prompt_trace["parse_error_stage"] = "langchain_parser_non_dict"
        return {}

    @staticmethod
    def _extract_candidates(parsed: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            valid = ModelIntentResponseSchema.model_validate(parsed)
            return [item.model_dump() for item in valid.candidates]
        except ValidationError:
            return []

    @staticmethod
    def _build_response_schema_instructions() -> str:
        return (
            "Верни только JSON-объект без пояснений и без markdown.\n"
            "Строгая схема ответа:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"intent": "<код_темы_из_списка>", "score": 0.0, "evidence": "<краткая причина>"}\n'
            "  ]\n"
            "}\n"
            "Ограничения:\n"
            "- В candidates добавляй 1 или 2 элемента.\n"
            "- intent только из списка тем.\n"
            "- score в диапазоне [0..1].\n"
            "- Никакого текста до или после JSON."
        )

    @staticmethod
    def _build_retry_prompt(base_prompt: str) -> str:
        return (
            f"{base_prompt}\n"
            "Повторите ответ строго в JSON формате по схеме. "
            "Никакого текста до или после JSON."
        )

    def _normalize_candidates(self, candidates: Any) -> list[ModelIntentScore]:
        if not isinstance(candidates, list):
            return []
        intent_set = set(self._intents)
        normalized_by_intent: dict[str, ModelIntentScore] = {}
        for item in candidates:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip()
            if not intent:
                continue
            if intent not in intent_set:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            evidence = str(item.get("evidence", "")).strip()
            if evidence:
                evidence = " ".join(evidence.split()[: self._max_evidence_tokens])
            candidate = ModelIntentScore(
                intent=intent,
                score=max(0.0, min(1.0, score)),
                evidence=evidence,
            )
            existing = normalized_by_intent.get(intent)
            if existing is None or candidate.score > existing.score:
                normalized_by_intent[intent] = candidate
        normalized = sorted(normalized_by_intent.values(), key=lambda row: row.score, reverse=True)
        return normalized
