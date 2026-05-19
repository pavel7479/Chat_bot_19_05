from __future__ import annotations

from src.retrieval.fact_repository import FactRepository


class ServiceResponseProgressPolicy:
    def select_next_fact_ids(
        self,
        *,
        candidate_fact_ids: list[str],
        used_fact_ids: list[str],
        used_semantic_groups: list[str],
        fact_catalog: FactRepository,
    ) -> list[str]:
        used_ids = {str(item).strip() for item in used_fact_ids if str(item).strip()}
        used_groups = {str(item).strip() for item in used_semantic_groups if str(item).strip()}

        for fact_id in candidate_fact_ids:
            fact = fact_catalog.get_by_fact_id(fact_id)
            if fact is None:
                continue
            if fact.fact_id in used_ids:
                continue
            semantic_group = str(fact.semantic_group).strip()
            if semantic_group and semantic_group in used_groups:
                continue
            return [fact.fact_id]

        for fact_id in candidate_fact_ids:
            fact = fact_catalog.get_by_fact_id(fact_id)
            if fact is None:
                continue
            semantic_group = str(fact.semantic_group).strip()
            if semantic_group and semantic_group in used_groups:
                continue
            return [fact.fact_id]

        return [candidate_fact_ids[0]] if candidate_fact_ids else []
