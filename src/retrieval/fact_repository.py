from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.core.models import FactRecord


class FactRepository:
    _SAFE_TOPIC_FALLBACK_ACTIONS = {
        "",
        "clarify_request",
        "out_of_scope_response",
    }

    def __init__(self, facts_file_path: Path) -> None:
        self._facts_file_path = facts_file_path
        self._facts = self._load()

    def list_facts(self) -> list[FactRecord]:
        return list(self._facts)

    def get_by_fact_id(self, fact_id: str) -> FactRecord | None:
        wanted = str(fact_id).strip()
        if not wanted:
            return None
        for fact in self._facts:
            if fact.fact_id == wanted:
                return fact
        return None

    def find_by_ids(self, fact_ids: list[str]) -> list[FactRecord]:
        ordered: list[FactRecord] = []
        for fact_id in fact_ids:
            fact = self.get_by_fact_id(fact_id)
            if fact is not None:
                ordered.append(fact)
        return ordered

    def _load(self) -> list[FactRecord]:
        if not self._facts_file_path.exists():
            return []
        raw = yaml.safe_load(self._facts_file_path.read_text(encoding="utf-8")) or {}
        facts: list[FactRecord] = []
        for item in raw.get("facts", []):
            if not isinstance(item, dict):
                continue
            fact_id = str(item.get("fact_id", "")).strip()
            topic = str(item.get("topic", "")).strip()
            text = str(item.get("text", "")).strip()
            template = str(item.get("template", "")).strip()
            if not fact_id or not topic or (not text and not template):
                continue
            facts.append(
                FactRecord(
                    fact_id=fact_id,
                    topic=topic,
                    subtopic=str(item.get("subtopic", "")).strip(),
                    entity_type=str(item.get("entity_type", "")).strip(),
                    entity=str(item.get("entity", "")).strip(),
                    text=text,
                    fact_type=str(item.get("fact_type", "knowledge")).strip().lower() or "knowledge",
                    section_tag=str(item.get("section_tag", "general")).strip().lower() or "general",
                    priority=int(item.get("priority", 0)),
                    aliases=[str(alias).strip().lower() for alias in item.get("aliases", []) if str(alias).strip()],
                    action_tags=[str(tag).strip() for tag in item.get("action_tags", []) if str(tag).strip()],
                    template=template,
                    required_slots=[str(slot).strip() for slot in item.get("required_slots", []) if str(slot).strip()],
                    semantic_group=str(item.get("semantic_group", "")).strip(),
                )
            )
        return facts

    def find_best(self, topic_ids: list[str], action_name: str, user_query: str, history_text: str) -> FactRecord | None:
        if not self._facts:
            return None
        query = self._normalize(user_query)
        history = self._normalize(history_text)
        topic_set = set(topic_ids)
        action_candidates = [
            fact
            for fact in self._facts
            if action_name and action_name in fact.action_tags
        ]
        # Strict action-tag filtering by default.
        if action_candidates:
            candidate_pool = action_candidates
        elif action_name in self._SAFE_TOPIC_FALLBACK_ACTIONS:
            candidate_pool = [fact for fact in self._facts if fact.topic in topic_set]
        else:
            return None

        if not candidate_pool:
            return None
        candidates: list[tuple[int, FactRecord]] = []
        for fact in candidate_pool:
            if fact.fact_type != "knowledge":
                continue
            score = 0
            topic_match = fact.topic in topic_set
            action_match = bool(action_name and action_name in fact.action_tags)
            # Guardrail: do not pick facts from unrelated actions/topics only by history aliases.
            if not (topic_match or action_match):
                continue
            if topic_match:
                score += 8
            if action_match:
                score += 12
            if fact.subtopic and fact.subtopic in topic_set:
                score += 3
            query_alias_hits = sum(1 for alias in fact.aliases if alias and alias in query)
            history_alias_hits = sum(1 for alias in fact.aliases if alias and alias in history)
            # Current user query must dominate over historical context.
            score += query_alias_hits * 8
            score += history_alias_hits * 2
            if score <= 0:
                continue
            score += fact.priority
            candidates.append((score, fact))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()
