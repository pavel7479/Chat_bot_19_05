from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.agents.response_planner import ResponsePlan
from src.agents.policy.types import ResponseAction, ResponseState
from src.agents.response_services.action_compatibility_validator import (
    ActionCompatibilityValidator,
    ActionValidationResult,
)
from src.agents.response_services.meaning_ownership_policy import MeaningOwnershipPolicy
from src.agents.response_services.meaning_validator import MeaningValidationResult, MeaningValidator
from src.app.price_provider import PriceContext
from src.core.models import (
    BotResponse,
    EvidenceItem,
    EvidencePack,
    MandatoryMeaningBlock,
    PreparedResponseContext,
    RetrievedChunk,
    TopicClassificationResult,
)


@dataclass(slots=True)
class AnswerBuildResult:
    answer_prompt: str
    answer_text: str
    raw_answer: str
    reasoning_summary: str
    selection_trace: dict[str, object]
    mandatory_blocks: list[MandatoryMeaningBlock]


class GroundedAnswerComposer:
    """Compose deterministic canonical answer from prepared facts and pricing state."""

    def compose(
        self,
        action_name: str,
        response_plan: list[str],
        chunks: list[RetrievedChunk],
        fallback_answer: str,
        prepared_context: PreparedResponseContext,
        price_context: PriceContext,
        mandatory_blocks: list[MandatoryMeaningBlock],
    ) -> str:
        if price_context.price_lines:
            pricing_answer = self._compose_pricing(price_context=price_context, prepared_context=prepared_context)
            if pricing_answer:
                return pricing_answer
        if prepared_context.primary_facts:
            return "\n".join(prepared_context.primary_facts[:3]).strip() or fallback_answer
        parts = self._extract_parts(chunks)
        if not parts:
            return fallback_answer
        ordered = self._order_parts(parts, response_plan)
        if not ordered:
            return fallback_answer
        return "\n".join(ordered[:3]).strip() or fallback_answer

    @staticmethod
    def _compose_pricing(*, price_context: PriceContext, prepared_context: PreparedResponseContext) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for line in price_context.price_lines:
            text = str(line).strip()
            if text and text not in seen:
                lines.append(text)
                seen.add(text)
        if not lines:
            for line in prepared_context.prices:
                text = str(line).strip()
                if text and text not in seen:
                    lines.append(text)
                    seen.add(text)
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_parts(chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        for chunk in chunks[:4]:
            text = chunk.text.strip()
            section = (
                str(chunk.metadata.get("section_tag", "general")).strip()
                if isinstance(chunk.metadata, dict)
                else "general"
            )
            if not text:
                continue
            for line in text.splitlines():
                line = line.strip()
                if len(line) < 20:
                    continue
                item = (section, line)
                if item not in parts:
                    parts.append(item)
        return parts

    @staticmethod
    def _order_parts(parts: list[tuple[str, str]], response_plan: list[str]) -> list[str]:
        if not response_plan:
            return [text for _, text in parts]
        part_sections = {section for section, _ in parts}
        if not part_sections.intersection(response_plan):
            return [text for _, text in parts]
        plan_weight = {name: float(len(response_plan) - idx) for idx, name in enumerate(response_plan)}
        scored: list[tuple[float, str]] = []
        for section, text in parts:
            score = plan_weight.get(section, 0.0)
            scored.append((score, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]


class AnswerSynthesisService:
    def __init__(
        self,
        meaning_policy: MeaningOwnershipPolicy | None = None,
        meaning_validator: MeaningValidator | None = None,
        action_validator: ActionCompatibilityValidator | None = None,
    ) -> None:
        self._meaning_policy = meaning_policy or MeaningOwnershipPolicy()
        self._meaning_validator = meaning_validator or MeaningValidator()
        self._action_validator = action_validator or ActionCompatibilityValidator()

    def build_answer(
        self,
        llm_provider: Any,
        prompt_manager: Any,
        composer: Any,
        validator: Any,
        anti_repeat_policy: Any,
        history_text: str,
        user_query: str,
        topic_result: TopicClassificationResult,
        response_state: ResponseState,
        selected_action: ResponseAction,
        response_plan: ResponsePlan,
        effective_chunks: list[RetrievedChunk],
        retrieved_context: str,
        evidence_reason: str,
        answerability_status: str,
    ) -> AnswerBuildResult:
        effective_context = "\n\n".join(chunk.text for chunk in effective_chunks)
        first_agent_output = (
            topic_result.diagnostics.get("first_agent_output", {})
            if isinstance(topic_result.diagnostics, dict)
            else {}
        )
        first_agent_data = (
            topic_result.diagnostics.get("first_agent_data", {})
            if isinstance(topic_result.diagnostics, dict)
            else {}
        )
        answer_block = first_agent_data.get("answer", {}) if isinstance(first_agent_data, dict) else {}
        if not isinstance(answer_block, dict):
            answer_block = {}
        structured_context = self._extract_structured_context(answer_block)
        price_context = self._extract_price_context(answer_block)

        rewritten_query = self._extract_rewritten_query(first_agent_output, user_query)
        intents_with_scores = self._extract_intents_with_scores(first_agent_output, topic_result)
        normalized_brands = self._extract_normalized_brands(first_agent_data)
        dialog_text = self._to_client_bot_dialog(history_text=history_text, user_query=user_query)
        answer_prompt = prompt_manager.build_answer_prompt(
            dialog_text=dialog_text,
            rewritten_query=rewritten_query,
            intents_with_scores=intents_with_scores,
            normalized_brands=normalized_brands,
            retrieved_facts_text=retrieved_context or effective_context,
            user_query=user_query,
        )
        fallback_answer = self._fallback_answer_from_context(retrieved_context or effective_context)
        mandatory_blocks = self._meaning_policy.build(
            selected_action=selected_action,
            response_plan=response_plan,
            price_context=price_context,
            structured_context=structured_context,
        )
        deterministic_answer = composer.compose(
            action_name=selected_action.name,
            response_plan=list(response_plan.required_fact_ids) + list(response_plan.required_price_blocks),
            chunks=effective_chunks,
            fallback_answer=fallback_answer,
            prepared_context=structured_context,
            price_context=price_context,
            mandatory_blocks=mandatory_blocks,
        )
        answer_text, raw_answer, selection_trace = self._generate_answer_text(
            llm_provider=llm_provider,
            answer_prompt=answer_prompt,
            selected_action=selected_action,
            price_context=price_context,
            deterministic_answer=deterministic_answer,
            fallback_answer=fallback_answer,
            mandatory_blocks=mandatory_blocks,
            answerability_status=answerability_status,
            force_deterministic_service=(
                selected_action.name == "company_services"
                and (
                    response_state.last_action_name == "company_services"
                    or response_state.active_business_flow == "company_services"
                )
            ),
        )
        answer_text = anti_repeat_policy.apply(
            action=selected_action,
            answer_text=answer_text,
            history_text=history_text,
            user_query=user_query,
        )
        if not raw_answer:
            raw_answer = json.dumps({"answer_text": answer_text}, ensure_ascii=False)
        reasoning_summary = (
            f"Детерминированное действие: {selected_action.name}. "
            f"Темы: {topic_result.topic_ids}. "
            f"evidence={evidence_reason}; hits={len(effective_chunks)}"
        )
        if isinstance(topic_result.diagnostics, dict):
            topic_result.diagnostics["answer_selection_trace"] = dict(selection_trace)
            topic_result.diagnostics["mandatory_meaning_blocks_trace"] = {
                "mandatory_blocks": [block.as_dict() for block in mandatory_blocks],
                "final_answer_source": str(selection_trace.get("selected_source", "")),
                "llm_validation": dict(selection_trace.get("llm_validation", {})),
            }
        return AnswerBuildResult(
            answer_prompt=answer_prompt,
            answer_text=answer_text,
            raw_answer=raw_answer,
            reasoning_summary=reasoning_summary,
            selection_trace=selection_trace,
            mandatory_blocks=mandatory_blocks,
        )

    def _generate_answer_text(
        self,
        llm_provider: Any,
        answer_prompt: str,
        selected_action: ResponseAction,
        price_context: PriceContext,
        deterministic_answer: str,
        fallback_answer: str,
        mandatory_blocks: list[MandatoryMeaningBlock],
        answerability_status: str,
        force_deterministic_service: bool = False,
    ) -> tuple[str, str, dict[str, object]]:
        raw_answer = ""
        llm_answer = ""
        action_result = ActionValidationResult(is_valid=False)
        meaning_result = MeaningValidationResult(is_valid=False, block_results={})

        try:
            raw_answer = str(llm_provider.generate_json(answer_prompt) or "").strip()
            llm_answer = self._parse_answer_text(raw_answer)
            action_result = (
                self._action_validator.validate(selected_action, llm_answer)
                if llm_answer
                else ActionValidationResult(is_valid=False)
            )
            if action_result.is_valid:
                meaning_result = self._meaning_validator.validate(llm_answer, mandatory_blocks)
        except Exception:
            llm_answer = ""
            action_result = ActionValidationResult(is_valid=False)
            meaning_result = MeaningValidationResult(is_valid=False, block_results={})

        selected_text = deterministic_answer or fallback_answer or self._safe_default_answer(selected_action)
        selected_source = "deterministic" if deterministic_answer else ("fallback" if fallback_answer else "safe_default")
        selection_reason = "Used deterministic answer as business source of truth."
        hard_deterministic_pricing = (
            price_context.tis_price_status == "missing"
            or (
                selected_action.name == "tis_tariffs"
                and (
                    bool(price_context.unknown_brand_mentions)
                    or bool(price_context.missing_tis_price_brands)
                    or len(price_context.recognized_brands) > 1
                )
            )
        )
        hard_deterministic_service = force_deterministic_service
        hard_locked_action = bool(selected_action.locked_action)

        if llm_answer and action_result.is_valid and meaning_result.is_valid and not hard_deterministic_pricing and not hard_deterministic_service and not hard_locked_action:
            selected_text = llm_answer
            selected_source = "llm"
            selection_reason = "LLM answer preserved mandatory meaning and action compatibility."
        elif hard_locked_action:
            selected_text = deterministic_answer or fallback_answer or self._safe_default_answer(selected_action)
            selected_source = "deterministic" if deterministic_answer else ("fallback" if fallback_answer else "safe_default")
            selection_reason = "Locked business action enforced deterministic answer ownership."
        elif hard_deterministic_pricing:
            selected_text = deterministic_answer or fallback_answer or self._safe_default_answer(selected_action)
            selected_source = "deterministic" if deterministic_answer else ("fallback" if fallback_answer else "safe_default")
            selection_reason = "Hard deterministic pricing policy applied for missing TIS price state."
        elif hard_deterministic_service:
            selected_text = deterministic_answer or fallback_answer or self._safe_default_answer(selected_action)
            selected_source = "deterministic" if deterministic_answer else ("fallback" if fallback_answer else "safe_default")
            selection_reason = "Deterministic service progression preserved anti-repeat semantics."
        elif not deterministic_answer:
            if answerability_status != "ok" and fallback_answer and self._fallback_is_compatible(selected_action, fallback_answer):
                selected_text = fallback_answer
                selected_source = "fallback"
                selection_reason = "Deterministic answer empty; used compatible fallback answer."
            elif not selected_text:
                selected_text = self._safe_default_answer(selected_action)
                selected_source = "safe_default"
                selection_reason = "No deterministic or compatible fallback answer remained; used safe default."
            elif llm_answer and not action_result.is_valid:
                selection_reason = "Rejected LLM answer due to action incompatibility; used fallback/default."
            elif llm_answer:
                selection_reason = "Rejected LLM answer due to meaning validation failure; used fallback/default."

        trace = {
            "llm_answer": llm_answer,
            "fallback_answer": fallback_answer,
            "deterministic_answer": deterministic_answer,
            "validator_passed": bool(llm_answer) and action_result.is_valid and meaning_result.is_valid,
            "selected_source": selected_source,
            "selection_reason": selection_reason,
            "mandatory_blocks": [block.as_dict() for block in mandatory_blocks],
            "llm_validation": dict(meaning_result.block_results),
            "action_compatible": action_result.is_valid,
            "validator_trace": action_result.as_dict(),
            "final_answer_source": selected_source,
            "hard_deterministic_pricing": hard_deterministic_pricing,
            "hard_deterministic_service": hard_deterministic_service,
            "hard_locked_action": hard_locked_action,
        }
        return selected_text, raw_answer, trace

    @staticmethod
    def _extract_structured_context(answer_block: dict[str, object]) -> PreparedResponseContext:
        raw = answer_block.get("structured_context", {}) if isinstance(answer_block, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        return PreparedResponseContext(
            primary_facts=list(raw.get("primary_facts", [])) if isinstance(raw.get("primary_facts", []), list) else [],
            secondary_facts=list(raw.get("secondary_facts", [])) if isinstance(raw.get("secondary_facts", []), list) else [],
            prices=list(raw.get("prices", [])) if isinstance(raw.get("prices", []), list) else [],
            followup_questions=list(raw.get("followup_questions", [])) if isinstance(raw.get("followup_questions", []), list) else [],
            slots=dict(raw.get("slots", {})) if isinstance(raw.get("slots", {}), dict) else {},
            product_context=dict(raw.get("product_context", {})) if isinstance(raw.get("product_context", {}), dict) else {},
        )

    @staticmethod
    def _extract_price_context(answer_block: dict[str, object]) -> PriceContext:
        raw = answer_block.get("price_context", {}) if isinstance(answer_block, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        return PriceContext(
            product=str(raw.get("product", "")).strip(),
            brands=list(raw.get("brands", [])) if isinstance(raw.get("brands", []), list) else [],
            recognized_brands=list(raw.get("recognized_brands", [])) if isinstance(raw.get("recognized_brands", []), list) else [],
            unknown_brand_mentions=list(raw.get("unknown_brand_mentions", [])) if isinstance(raw.get("unknown_brand_mentions", []), list) else [],
            priced_brands=list(raw.get("priced_brands", [])) if isinstance(raw.get("priced_brands", []), list) else [],
            price_lines=list(raw.get("price_lines", [])) if isinstance(raw.get("price_lines", []), list) else [],
            evidence_items=list(raw.get("evidence_items", [])) if isinstance(raw.get("evidence_items", []), list) else [],
            missing_tis_price_brands=list(raw.get("missing_tis_price_brands", [])) if isinstance(raw.get("missing_tis_price_brands", []), list) else [],
            tis_price_status=str(raw.get("tis_price_status", "not_requested")).strip() or "not_requested",
            fallback_price_blocks=list(raw.get("fallback_price_blocks", [])) if isinstance(raw.get("fallback_price_blocks", []), list) else [],
            pricing_mode=str(raw.get("pricing_mode", "all")).strip() or "all",
        )

    @staticmethod
    def _safe_default_answer(selected_action: ResponseAction) -> str:
        defaults = {
            "company_services": "Могу рассказать о возможностях сервиса, каталогах и сценариях работы для автобизнеса.",
            "human_operator": "Подключу менеджера: он поможет разобрать вопрос и свяжется с вами по дальнейшим шагам.",
            "clarify_request": "Уточните, пожалуйста, что именно вас интересует.",
        }
        return defaults.get(selected_action.name, "Уточните, пожалуйста, что именно вас интересует.")

    @staticmethod
    def _parse_answer_text(raw_answer: str) -> str:
        if not raw_answer:
            return ""
        parsed = AnswerSynthesisService._safe_load_json_object(raw_answer)
        if not isinstance(parsed, dict):
            return ""
        return str(parsed.get("answer_text", "")).strip()

    def _fallback_is_compatible(self, selected_action: ResponseAction, answer_text: str) -> bool:
        return self._action_validator.validate(selected_action, answer_text).is_valid

    @staticmethod
    def _safe_load_json_object(raw_text: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(raw_text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

    @staticmethod
    def _extract_rewritten_query(first_agent_output: dict[str, object], user_query: str) -> str:
        if isinstance(first_agent_output, dict):
            value = str(first_agent_output.get("rewritten_query", "")).strip()
            if value:
                return value
        return user_query

    @staticmethod
    def _extract_intents_with_scores(first_agent_output: dict[str, object], topic_result: TopicClassificationResult) -> str:
        if isinstance(first_agent_output, dict):
            scores = first_agent_output.get("intent_scores", [])
            if isinstance(scores, list) and scores:
                formatted: list[str] = []
                for row in scores:
                    if not isinstance(row, dict):
                        continue
                    intent = str(row.get("intent", "")).strip()
                    try:
                        score = float(row.get("score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    if intent:
                        formatted.append(f"{intent}: {score:.2f}")
                if formatted:
                    return "; ".join(formatted)
        return f"{topic_result.primary_topic_id}: {float(topic_result.confidence):.2f}"

    @staticmethod
    def _extract_normalized_brands(first_agent_data: dict[str, object]) -> str:
        if isinstance(first_agent_data, dict):
            brands = first_agent_data.get("brands", {})
            if isinstance(brands, dict):
                normalized = brands.get("normalized", [])
                if isinstance(normalized, list):
                    values = [str(item).strip() for item in normalized if str(item).strip()]
                    if values:
                        return ", ".join(values)
        return "-"

    @staticmethod
    def _to_client_bot_dialog(history_text: str, user_query: str) -> str:
        lines: list[str] = []
        for raw in str(history_text).splitlines():
            row = raw.strip()
            if not row:
                continue
            if row.lower().startswith("user:"):
                lines.append(f"клиент: {row.split(':', 1)[1].strip()}")
            elif row.lower().startswith("assistant:"):
                lines.append(f"бот: {row.split(':', 1)[1].strip()}")
            else:
                lines.append(row)
        last_client = ""
        for line in reversed(lines):
            if line.lower().startswith("клиент:"):
                last_client = line.split(":", 1)[1].strip()
                break
        current_query = str(user_query or "").strip()
        if current_query and last_client != current_query:
            lines.append(f"клиент: {current_query}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer_from_context(context_text: str) -> str:
        lines = AnswerSynthesisService._meaningful_context_lines(context_text)
        if not lines:
            return ""
        return "\n".join(lines[:2]).strip()

    @staticmethod
    def _use_alternative_context_line_if_repeated(
        answer_text: str,
        history_text: str,
        context_text: str,
    ) -> str:
        normalized_answer = AnswerSynthesisService._normalize_text(answer_text)
        if not normalized_answer:
            return answer_text
        recent_answers = AnswerSynthesisService._recent_assistant_answers(history_text, limit=2)
        recent_normalized = {AnswerSynthesisService._normalize_text(item) for item in recent_answers}
        if normalized_answer not in recent_normalized:
            return answer_text
        for candidate in AnswerSynthesisService._meaningful_context_lines(context_text):
            normalized_candidate = AnswerSynthesisService._normalize_text(candidate)
            if not normalized_candidate:
                continue
            if normalized_candidate not in recent_normalized and normalized_candidate != normalized_answer:
                return candidate
        return answer_text

    @staticmethod
    def _meaningful_context_lines(context_text: str) -> list[str]:
        lines: list[str] = []
        ignored_headers = {
            "facts:",
            "extra facts:",
            "prices:",
            "followup:",
            "нормализованные бренды:",
        }
        for raw in str(context_text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered in ignored_headers:
                continue
            if lowered.startswith("нормализованные бренды:"):
                continue
            if line not in lines:
                lines.append(line)
        if lines:
            return lines
        fallback = str(context_text or "").strip()
        return [fallback] if fallback else []

    @staticmethod
    def _recent_assistant_answers(history_text: str, limit: int = 2) -> list[str]:
        lines = [line.strip() for line in str(history_text).splitlines() if line.strip()]
        answers: list[str] = []
        current: list[str] = []
        collecting = False
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("user:"):
                if collecting and current:
                    answers.append("\n".join(reversed(current)).strip())
                    current = []
                    collecting = False
                continue
            if lowered.startswith("assistant:"):
                text = line.split(":", 1)[1].strip()
                if text:
                    current.append(text)
                answers.append("\n".join(reversed(current)).strip())
                current = []
                collecting = False
                if len(answers) >= limit:
                    break
                continue
            current.append(line)
            collecting = True
        return answers

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()


class ResponseModelFactory:
    def build(
        self,
        topic_result: TopicClassificationResult,
        selected_action: ResponseAction,
        response_plan: ResponsePlan,
        effective_chunks: list[RetrievedChunk],
        answer_text: str,
        reasoning_summary: str,
    ) -> BotResponse:
        retrieval_trace = topic_result.retrieval_context if isinstance(topic_result.retrieval_context, dict) else {}
        trace_complete = self._is_trace_complete(retrieval_trace)
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            first_agent_data = {}
        answer_block = first_agent_data.get("answer", {})
        if not isinstance(answer_block, dict):
            answer_block = {}
        answer_evidence_items = answer_block.get("evidence_items", [])
        if not isinstance(answer_evidence_items, list):
            answer_evidence_items = []
        used_evidence_ids = answer_block.get("used_evidence_ids", [])
        if not isinstance(used_evidence_ids, list):
            used_evidence_ids = []

        if answer_evidence_items:
            evidence_items = [
                EvidenceItem(
                    evidence_id=str(item.get("evidence_id", "")).strip(),
                    text=str(item.get("text", "")).strip(),
                    score=float(item.get("score", 0.0) or 0.0),
                    source=str(item.get("source", "")).strip() or "knowledge",
                    action_name=selected_action.name,
                    why_selected="prepared_context",
                    section_tag=str(item.get("section_tag", "general")).strip() or "general",
                    source_scores={
                        "bm25": 0.0,
                        "dense": 0.0,
                        "final": float(item.get("score", 0.0) or 0.0),
                    },
                )
                for item in answer_evidence_items
                if str(item.get("evidence_id", "")).strip() and str(item.get("text", "")).strip()
            ]
        else:
            evidence_items = [
                EvidenceItem(
                    evidence_id=chunk.metadata.get("entry_index", ""),
                    text=chunk.text,
                    score=chunk.score,
                    source=chunk.source,
                    action_name=selected_action.name,
                    why_selected=str(chunk.metadata.get("why_selected", "")),
                    section_tag=str(chunk.metadata.get("section_tag", "general")),
                    source_scores={
                        "bm25": float(chunk.metadata.get("bm25_score", 0.0) or 0.0),
                        "dense": float(chunk.metadata.get("dense_score", 0.0) or 0.0),
                        "final": float(chunk.score),
                    },
                )
                for chunk in effective_chunks
            ]

        evidence_pack = EvidencePack(
            items=evidence_items,
            status=reasoning_summary.split("evidence=")[-1].split(";")[0] if "evidence=" in reasoning_summary else "unknown",
        )
        answer_sections = list(response_plan.required_fact_ids) + list(response_plan.required_price_blocks)
        if not answer_sections and response_plan.primary_action:
            answer_sections = [response_plan.primary_action]
        return BotResponse(
            answer_text=answer_text,
            reasoning_summary=reasoning_summary,
            topic_id=topic_result.primary_topic_id,
            topic_ids=topic_result.topic_ids,
            sources=[item.source for item in evidence_items],
            action_name=selected_action.name,
            planned_action=selected_action.name,
            used_evidence_ids=[
                str(item).strip()
                for item in used_evidence_ids
                if str(item).strip()
            ] or [
                item.evidence_id
                for item in evidence_items
                if item.evidence_id
            ],
            answer_sections=answer_sections,
            contract_flags={
                "clarify_required": bool(topic_result.clarify_required),
                "nonsense_input": bool(topic_result.nonsense_input),
                "abuse_input": bool(topic_result.abuse_input),
                "planned_action_matches": True,
                "trace_complete": trace_complete,
            },
            media_refs=[],
            evidence_pack=evidence_pack,
        )

    @staticmethod
    def _is_trace_complete(trace: dict[str, object]) -> bool:
        legacy_keys = {
            "trace_id",
            "query_received",
            "query_extended",
            "bm25_hits",
            "dense_hits",
            "merged_candidates",
            "reranked_topk",
            "dropped_candidates",
        }
        deterministic_keys = {
            "trace_id",
            "selection_mode",
            "query_received",
            "selected_fact_ids",
            "price_blocks",
            "chunks",
        }
        keys = set(trace.keys())
        return legacy_keys.issubset(keys) or deterministic_keys.issubset(keys)
