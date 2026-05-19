from __future__ import annotations

from dataclasses import dataclass

from src.core.diff_utils import dict_diff, snapshot
from src.core.models import TopicClassificationResult
from src.domain.brands import BrandAliasResolver as DomainBrandAliasResolver


@dataclass(slots=True)
class TurnBrandAliasResolver:
    """Per-turn brand alias normalization and diagnostics enrichment."""

    brand_resolver: DomainBrandAliasResolver

    def enrich(self, topic_result: TopicClassificationResult, user_query: str) -> None:
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        first_agent_data = diagnostics.get("first_agent_data", {})
        if not isinstance(first_agent_data, dict):
            first_agent_data = {}
            diagnostics["first_agent_data"] = first_agent_data

        first_agent_output = diagnostics.get("first_agent_output", {})
        if not isinstance(first_agent_output, dict):
            first_agent_output = {}
        query = str(first_agent_output.get("rewritten_query") or user_query)
        aliases = self.brand_resolver.extract(query)
        if not aliases:
            state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
            inherited = str(state_snapshot.get("last_mentioned_brand", "")).strip().lower()
            if inherited:
                aliases = [inherited]
        normalized = sorted({item.lower() for item in aliases if str(item).strip()})

        brands_block = first_agent_data.get("brands", {})
        if not isinstance(brands_block, dict):
            brands_block = {}
        before_brands = snapshot(brands_block)
        brands_block["detected"] = list(aliases)
        brands_block["normalized"] = list(normalized)
        first_agent_data["brands"] = brands_block
        after_brands = snapshot(first_agent_data["brands"])

        trace = diagnostics.get("first_agent_trace", {})
        if not isinstance(trace, dict):
            trace = {}
            diagnostics["first_agent_trace"] = trace
        pipeline_steps = trace.get("pipeline_steps", [])
        if not isinstance(pipeline_steps, list):
            pipeline_steps = []
            trace["pipeline_steps"] = pipeline_steps
        pipeline_steps.append(
            {
                "step": "brand_alias_resolver",
                "actor": "TurnBrandAliasResolver",
                "target": "first_agent_data.brands",
                "status": "ok",
                "before": before_brands,
                "after": after_brands,
                "diff": dict_diff(before_brands, after_brands),
            }
        )
        topic_result.diagnostics = diagnostics
