from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.response_planner import ResponsePlan
from src.app.fact_template_renderer import FactRenderResult, FactTemplateRenderer
from src.core.diff_utils import dict_diff, snapshot
from src.core.models import FactRecord, TopicClassificationResult
from src.domain.brand_display_resolver import BrandDisplayResolver
from src.retrieval.fact_repository import FactRepository


@dataclass(slots=True)
class KnowledgeRetriever:
    """Intent/topic-based fact extraction from facts.yaml (no model calls)."""

    fact_repository: FactRepository
    max_facts_per_turn: int = 8
    fact_renderer: FactTemplateRenderer = field(default_factory=FactTemplateRenderer)

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
        trace = diagnostics.get("first_agent_trace", {})
        if not isinstance(trace, dict):
            trace = {}
        diagnostics["first_agent_trace"] = trace
        pipeline_steps = trace.get("pipeline_steps", [])
        if not isinstance(pipeline_steps, list):
            pipeline_steps = []
            trace["pipeline_steps"] = pipeline_steps

        facts = self._select_facts(
            topic_ids=list(topic_result.topic_ids),
            user_query=user_query,
            history_text=history_text,
            response_plan=response_plan,
        )
        render_slots = self._build_render_slots(topic_result)
        retrieved_facts = []
        render_trace: list[dict[str, object]] = []
        for fact in facts:
            render_result = self.fact_renderer.render_with_trace(fact, render_slots)
            render_trace.append(dict(render_result.trace) | {"status": render_result.status})
            if not render_result.text:
                fallback_fact = self._resolve_render_fallback_fact(fact, render_result)
                if fallback_fact is None:
                    continue
                fallback_result = self.fact_renderer.render_with_trace(fallback_fact, render_slots)
                render_trace.append(dict(fallback_result.trace) | {"status": fallback_result.status, "fallback_for": fact.fact_id})
                if not fallback_result.text:
                    continue
                fact = fallback_fact
                render_result = fallback_result
            retrieved_facts.append(
                {
                    "fact_id": fact.fact_id,
                    "topic": fact.topic,
                    "subtopic": fact.subtopic,
                    "section_tag": fact.section_tag,
                    "semantic_group": fact.semantic_group,
                    "priority": fact.priority,
                    "text": render_result.text,
                }
            )
        retrieval_reason = [
            f"{fact.fact_id}:topic={fact.topic};action_tags={','.join(fact.action_tags) if fact.action_tags else '-'}"
            for fact in facts
        ]
        before_knowledge = snapshot(first_agent_data.get("knowledge", {}))

        first_agent_data["knowledge"] = {
            "selected_topics": list(topic_result.topic_ids),
            "selected_action": response_plan.primary_action if response_plan is not None else "",
            "selected_fact_ids": list(response_plan.required_fact_ids) if response_plan is not None else [],
            "retrieved_facts": retrieved_facts,
            "retrieval_reason": retrieval_reason,
            "render_trace": render_trace,
        }
        after_knowledge = snapshot(first_agent_data["knowledge"])
        pipeline_steps.append(
            {
                "step": "knowledge_retriever",
                "actor": "KnowledgeRetriever",
                "target": "first_agent_data.knowledge",
                "status": "ok",
                "before": before_knowledge,
                "after": after_knowledge,
                "diff": dict_diff(before_knowledge, after_knowledge),
            }
        )
        topic_result.diagnostics = diagnostics

    def _select_facts(
        self,
        topic_ids: list[str],
        user_query: str,
        history_text: str,
        response_plan: ResponsePlan | None = None,
    ) -> list[FactRecord]:
        if response_plan is not None and response_plan.required_fact_ids:
            return self.fact_repository.find_by_ids(response_plan.required_fact_ids)[: self.max_facts_per_turn]

        all_facts = self.fact_repository.list_facts()
        if not all_facts:
            return []

        ordered_topics = [topic for topic in topic_ids if topic]
        topic_order = {topic: idx for idx, topic in enumerate(ordered_topics)}
        topic_set = set(ordered_topics)
        scored: list[tuple[int, int, FactRecord]] = []
        for fact in all_facts:
            if fact.fact_type != "knowledge":
                continue
            if fact.topic not in topic_set:
                continue
            score = int(fact.priority)
            if fact.subtopic and fact.subtopic in topic_set:
                score += 3
            scored.append((topic_order.get(fact.topic, 999), -score, fact))

        scored.sort(key=lambda item: (item[0], item[1], item[2].fact_id))
        selected: list[FactRecord] = []
        seen_fact_ids: set[str] = set()
        for _, _, fact in scored:
            if fact.fact_id in seen_fact_ids:
                continue
            selected.append(fact)
            seen_fact_ids.add(fact.fact_id)
            if len(selected) >= self.max_facts_per_turn:
                break
        return selected

    @staticmethod
    def _build_render_slots(topic_result: TopicClassificationResult) -> dict[str, object]:
        display_resolver = BrandDisplayResolver()
        snapshot_state = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        state_slots = snapshot_state.get("slots", {})
        slots: dict[str, object] = dict(state_slots) if isinstance(state_slots, dict) else {}
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        slot_trace = diagnostics.get("slot_extraction_trace", {})
        if isinstance(slot_trace, dict):
            extracted_slots = slot_trace.get("slots", {})
            if isinstance(extracted_slots, dict):
                slots.update(extracted_slots)
        brand = str(slots.get("brand") or snapshot_state.get("last_mentioned_brand", "")).strip()
        if not brand:
            followup_trace = diagnostics.get("followup_trace", {})
            if isinstance(followup_trace, dict):
                brand = str(followup_trace.get("inherited_brand", "")).strip()
        if brand:
            slots["brand"] = brand.lower()
            slots["brand_display"] = display_resolver.display(str(slots.get("brand_display") or brand))
        return slots
