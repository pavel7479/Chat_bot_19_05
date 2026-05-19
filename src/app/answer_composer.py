from __future__ import annotations

from dataclasses import dataclass

from src.agents.response_planner import ResponsePlan
from src.app.pricing_flow_state_service import PricingFlowStateService
from src.core.diff_utils import dict_diff, snapshot
from src.core.models import PreparedResponseContext, SessionState
from src.core.models import TopicClassificationResult
from src.app.price_provider import PriceProvider


@dataclass(slots=True)
class AnswerComposer:
    """Prepare deterministic answer context from first-agent data only."""

    price_provider: PriceProvider | None = None
    pricing_flow_state_service: PricingFlowStateService | None = None
    max_facts_in_context: int = 5

    def enrich(
        self,
        topic_result: TopicClassificationResult,
        user_query: str,
        history_text: str,
        response_plan: ResponsePlan | None = None,
    ) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            first_agent_data = {}
        diagnostics["first_agent_data"] = first_agent_data

        knowledge = first_agent_data.get("knowledge", {})
        brands = first_agent_data.get("brands", {})
        answer_block = first_agent_data.get("answer", {})
        trace = diagnostics.get("first_agent_trace", {})

        if not isinstance(knowledge, dict):
            knowledge = {}
        if not isinstance(brands, dict):
            brands = {}
        if not isinstance(answer_block, dict):
            answer_block = {}
        if not isinstance(trace, dict):
            trace = {}
        diagnostics["first_agent_trace"] = trace
        pipeline_steps = trace.get("pipeline_steps", [])
        if not isinstance(pipeline_steps, list):
            pipeline_steps = []
            trace["pipeline_steps"] = pipeline_steps

        retrieved_facts = knowledge.get("retrieved_facts", [])
        if not isinstance(retrieved_facts, list):
            retrieved_facts = []
        merged_slots = self._merged_slots(topic_result)
        product_context = diagnostics.get("product_context_trace", {})
        if not isinstance(product_context, dict):
            product_context = {}
        pricing_mode = str(topic_result.state_snapshot.get("pricing_mode", "all")).strip() if isinstance(topic_result.state_snapshot, dict) else "all"
        pricing_flow = (
            self.pricing_flow_state_service.build_from_turn(
                session_state=SessionState(**topic_result.state_snapshot) if isinstance(topic_result.state_snapshot, dict) else SessionState(),
                merged_slots=merged_slots,
                pricing_mode=pricing_mode,
                product_hint=str(product_context.get("inferred_product_context", "tis")).strip() or "tis",
            )
            if self.pricing_flow_state_service is not None
            else None
        )
        normalized_brands = (
            list(pricing_flow.recognized_brands)
            if pricing_flow is not None and pricing_flow.recognized_brands
            else self._resolve_brands(topic_result=topic_result, brands_block=brands)
        )

        fact_texts: list[str] = []
        used_fact_ids: list[str] = []
        evidence_items: list[dict[str, object]] = []
        for fact in retrieved_facts:
            if not isinstance(fact, dict):
                continue
            text = str(fact.get("text", "")).strip()
            fact_id = str(fact.get("fact_id", "")).strip()
            if not text:
                continue
            fact_texts.append(text)
            if fact_id:
                used_fact_ids.append(fact_id)
                evidence_items.append(
                    {
                        "evidence_id": fact_id,
                        "text": text,
                        "score": float(fact.get("priority", 0) or 0.0),
                        "source": "facts.yaml",
                        "section_tag": str(fact.get("section_tag", "general")).strip() or "general",
                        "semantic_group": str(fact.get("semantic_group", "")).strip(),
                    }
                )
            if len(fact_texts) >= self.max_facts_in_context:
                break

        followup_trace = diagnostics.get("followup_trace", {})
        price_context = (
            self.price_provider.build_for_pricing_flow(
                required_price_blocks=list(response_plan.required_price_blocks) if response_plan is not None else [],
                pricing_flow=pricing_flow,
                followup_trace=followup_trace if isinstance(followup_trace, dict) else {},
            )
            if self.price_provider is not None and pricing_flow is not None
            else None
        )

        if price_context is not None:
            evidence_items.extend(price_context.evidence_items)
            diagnostics["product_context_trace"] = {
                **dict(diagnostics.get("product_context_trace", {})),
                "tis_price_status": price_context.tis_price_status,
                "fallback_price_blocks": list(price_context.fallback_price_blocks),
            }
            diagnostics["pricing_flow_trace"] = pricing_flow.as_dict() if pricing_flow is not None else {}
            product_context = diagnostics.get("product_context_trace", {})
            if not isinstance(product_context, dict):
                product_context = {}

        structured_context = PreparedResponseContext(
            primary_facts=list(fact_texts),
            secondary_facts=[],
            prices=list(price_context.lines) if price_context is not None else [],
            followup_questions=list(response_plan.required_followup) if response_plan is not None else [],
            slots=dict(merged_slots),
            product_context=dict(product_context),
        )
        prepared_context = self._render_prepared_context(
            structured_context=structured_context,
            normalized_brands=normalized_brands,
        )

        before_answer = snapshot(answer_block)
        answer_block["prepared_context"] = prepared_context
        answer_block["structured_context"] = structured_context.as_dict()
        answer_block["used_fact_ids"] = used_fact_ids
        answer_block["used_evidence_ids"] = [
            str(item.get("evidence_id", "")).strip()
            for item in evidence_items
            if str(item.get("evidence_id", "")).strip()
        ]
        answer_block["used_service_semantic_groups"] = [
            str(item.get("semantic_group", "")).strip()
            for item in evidence_items
            if str(item.get("semantic_group", "")).strip()
        ]
        answer_block["evidence_items"] = evidence_items
        if price_context is not None:
            answer_block["price_context"] = price_context.as_dict()
        first_agent_data["answer"] = answer_block
        after_answer = snapshot(first_agent_data["answer"])

        pipeline_steps.append(
            {
                "step": "answer_composer",
                "actor": "AnswerComposer",
                "target": "first_agent_data.answer",
                "status": "ok",
                "before": before_answer,
                "after": after_answer,
                "diff": dict_diff(before_answer, after_answer),
            }
        )
        topic_result.diagnostics = diagnostics

    @staticmethod
    def _render_prepared_context(
        structured_context: PreparedResponseContext,
        normalized_brands: list[object],
    ) -> str:
        parts: list[str] = []
        if normalized_brands:
            parts.append("НОРМАЛИЗОВАННЫЕ БРЕНДЫ:\n" + ", ".join(str(item) for item in normalized_brands))
        if structured_context.product_context:
            inferred = str(structured_context.product_context.get("inferred_product_context", "")).strip()
            if inferred:
                parts.append(f"PRODUCT CONTEXT:\n{inferred}")
        if structured_context.primary_facts:
            parts.append("FACTS:\n" + "\n".join(structured_context.primary_facts))
        if structured_context.secondary_facts:
            parts.append("EXTRA FACTS:\n" + "\n".join(structured_context.secondary_facts))
        if structured_context.prices:
            parts.append("PRICES:\n" + "\n".join(structured_context.prices))
        if structured_context.followup_questions:
            parts.append("FOLLOWUP:\n" + "\n".join(structured_context.followup_questions))
        return "\n\n".join(part for part in parts if str(part).strip())

    @staticmethod
    def _resolve_brands(
        topic_result: TopicClassificationResult,
        brands_block: dict[str, object],
    ) -> list[str]:
        normalized_brands = brands_block.get("normalized", [])
        if isinstance(normalized_brands, list):
            values = [str(item).strip().lower() for item in normalized_brands if str(item).strip()]
            if values:
                return values
        state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        pricing_mode = str(state_snapshot.get("pricing_mode", "all")).strip() or "all"
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        followup_trace = diagnostics.get("followup_trace", {})
        followup_type = str(followup_trace.get("followup_type", "")).strip() if isinstance(followup_trace, dict) else ""
        if pricing_mode in {"remaining_only", "explain_unresolved"} or followup_type in {"remaining_brands_followup", "clarification_pushback_followup"}:
            state_missing = state_snapshot.get("missing_price_brands", [])
            if isinstance(state_missing, list):
                values = [str(item).strip().lower() for item in state_missing if str(item).strip()]
                if values:
                    return values
            state_recognized = state_snapshot.get("recognized_brands", [])
            state_priced_raw = state_snapshot.get("priced_brands", [])
            state_priced = (
                {str(item).strip().lower() for item in state_priced_raw if str(item).strip()}
                if isinstance(state_priced_raw, list)
                else set()
            )
            if isinstance(state_recognized, list):
                values = [
                    str(item).strip().lower()
                    for item in state_recognized
                    if str(item).strip() and str(item).strip().lower() not in state_priced
                ]
                if values:
                    return values
        state_recognized = state_snapshot.get("recognized_brands", [])
        if isinstance(state_recognized, list):
            values = [str(item).strip().lower() for item in state_recognized if str(item).strip()]
            if values:
                return values
        if isinstance(followup_trace, dict) and followup_trace.get("is_followup"):
            inherited = str(followup_trace.get("inherited_brand", "")).strip().lower()
            if inherited:
                return [inherited]
        slot_brands = state_snapshot.get("slots", {}).get("brands", []) if isinstance(state_snapshot.get("slots", {}), dict) else []
        if isinstance(slot_brands, list):
            values = [str(item).strip().lower() for item in slot_brands if str(item).strip()]
            if values:
                return values
        last_brand = str(state_snapshot.get("last_mentioned_brand", "")).strip().lower()
        return [last_brand] if last_brand else []

    @staticmethod
    def _resolve_raw_brand_mentions(
        topic_result: TopicClassificationResult,
        merged_slots: dict[str, object],
    ) -> list[str]:
        raw_mentions = merged_slots.get("raw_brand_mentions", [])
        if isinstance(raw_mentions, list) and raw_mentions:
            return [str(item).strip() for item in raw_mentions if str(item).strip()]
        state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        requested = state_snapshot.get("requested_brands", [])
        if isinstance(requested, list):
            return [str(item).strip() for item in requested if str(item).strip()]
        return []

    @staticmethod
    def _resolve_unknown_brand_mentions(
        topic_result: TopicClassificationResult,
        merged_slots: dict[str, object],
    ) -> list[str]:
        unknown_mentions = merged_slots.get("unknown_brand_mentions", [])
        if isinstance(unknown_mentions, list) and unknown_mentions:
            return [str(item).strip() for item in unknown_mentions if str(item).strip()]
        state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        unknown = state_snapshot.get("unknown_brand_mentions", [])
        if isinstance(unknown, list):
            return [str(item).strip() for item in unknown if str(item).strip()]
        return []

    @staticmethod
    def _merged_slots(topic_result: TopicClassificationResult) -> dict[str, object]:
        state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        base_slots = state_snapshot.get("slots", {}) if isinstance(state_snapshot.get("slots", {}), dict) else {}
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        slot_trace = diagnostics.get("slot_extraction_trace", {}) if isinstance(diagnostics.get("slot_extraction_trace", {}), dict) else {}
        extracted_slots = slot_trace.get("slots", {}) if isinstance(slot_trace.get("slots", {}), dict) else {}
        merged = dict(base_slots)
        merged.update(extracted_slots)
        return merged
