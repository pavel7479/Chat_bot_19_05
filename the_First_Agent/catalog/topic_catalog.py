from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.core.models import TopicDefinition


class TopicCatalog:
    def __init__(self, topics_file: Path, max_examples_per_topic: int = 3) -> None:
        self._max_examples_per_topic = max(1, int(max_examples_per_topic))
        self._topics = self._load(topics_file)

    @staticmethod
    def _load(topics_file: Path) -> dict[str, TopicDefinition]:
        raw = yaml.safe_load(topics_file.read_text(encoding="utf-8")) or {}
        topics: dict[str, TopicDefinition] = {}
        topics_block = raw.get("topics")
        intents_block = raw.get("intents")

        if isinstance(topics_block, list):
            for item in topics_block:
                topic = TopicDefinition(
                    id=item["id"],
                    title=item["title"],
                    description=item["description"],
                    keywords=item.get("keywords", []),
                    aliases=item.get("aliases", []),
                    anti_keywords=item.get("anti_keywords", []),
                    choose_when="",
                    not_choose_when="",
                    choose_when_points=[],
                    not_choose_when_points=[],
                    examples=item.get("examples", []),
                )
                topics[topic.id] = topic
            return topics

        if isinstance(intents_block, list):
            for item in intents_block:
                if not isinstance(item, dict):
                    continue
                intent_id = str(item.get("intent", "")).strip()
                if not intent_id:
                    continue
                label_ru = str(item.get("label_ru", "")).strip() or intent_id
                choose_when_points = TopicCatalog._normalize_points(item.get("choose_when", ""))
                not_choose_when_points = TopicCatalog._normalize_points(item.get("not_choose_when", ""))
                examples = TopicCatalog._normalize_examples(item.get("examples", []))
                choose_when = "\n".join(choose_when_points)
                not_choose_when = "\n".join(not_choose_when_points)
                topic = TopicDefinition(
                    id=intent_id,
                    title=label_ru,
                    description=label_ru,
                    keywords=examples or TopicCatalog._extract_quoted_phrases(choose_when),
                    aliases=TopicCatalog._normalize_examples(item.get("aliases", [])),
                    anti_keywords=TopicCatalog._extract_quoted_phrases(not_choose_when),
                    choose_when=choose_when,
                    not_choose_when=not_choose_when,
                    choose_when_points=choose_when_points,
                    not_choose_when_points=not_choose_when_points,
                    examples=examples,
                )
                topics[topic.id] = topic
        return topics

    @staticmethod
    def _normalize_points(value: object) -> list[str]:
        if isinstance(value, list):
            points: list[str] = []
            for item in value:
                text = str(item).strip()
                if text and text not in points:
                    points.append(text)
            return points
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            return []
        chunks = re.split(r"(?<=[.!?])\s+|;\s+", text)
        points: list[str] = []
        for chunk in chunks:
            item = chunk.strip(" -")
            if item and item not in points:
                points.append(item)
        return points or [text]

    @staticmethod
    def _normalize_examples(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        examples: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in examples:
                examples.append(text)
        return examples

    @staticmethod
    def _extract_quoted_phrases(text: str) -> list[str]:
        if not text:
            return []
        phrases = re.findall(r'"([^"]+)"', text)
        unique: list[str] = []
        for phrase in phrases:
            normalized = phrase.strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique

    @property
    def topics(self) -> dict[str, TopicDefinition]:
        return self._topics

    def _selected_topics(self, topic_ids: list[str] | None = None) -> list[TopicDefinition]:
        if not topic_ids:
            return list(self._topics.values())
        selected: list[TopicDefinition] = []
        for topic_id in topic_ids:
            topic = self._topics.get(str(topic_id))
            if topic is not None:
                selected.append(topic)
        return selected

    def title_map(self, topic_ids: list[str] | None = None) -> dict[str, str]:
        return {topic.id: topic.title for topic in self._selected_topics(topic_ids)}

    def allowed_intents_text(self, topic_ids: list[str] | None = None) -> str:
        return "\n".join(f"- {topic.id}" for topic in self._selected_topics(topic_ids))

    def as_prompt_text(self, topic_ids: list[str] | None = None) -> str:
        blocks: list[str] = []
        selected_topics = self._selected_topics(topic_ids)
        selected_ids = {topic.id for topic in selected_topics}
        for topic in selected_topics:
            choose_when_lines = self._filter_points_for_shortlist(
                topic.choose_when_points or self._normalize_points(topic.choose_when or topic.description),
                selected_ids=selected_ids,
                current_topic_id=topic.id,
            )
            not_choose_when_lines = self._filter_points_for_shortlist(
                topic.not_choose_when_points or self._normalize_points(topic.not_choose_when or "-"),
                selected_ids=selected_ids,
                current_topic_id=topic.id,
            )
            examples = topic.examples[: self._max_examples_per_topic] or topic.keywords[: self._max_examples_per_topic]
            parts = [
                f"intent_id: {topic.id}",
                "choose_when:",
                *[f"- {line}" for line in choose_when_lines],
                "not_choose_when:",
                *[f"- {line}" for line in not_choose_when_lines],
            ]
            if examples:
                parts.extend(
                    [
                        "examples:",
                        *[f'- "{item}"' for item in examples],
                    ]
                )
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    def _filter_points_for_shortlist(
        self,
        points: list[str],
        *,
        selected_ids: set[str],
        current_topic_id: str,
    ) -> list[str]:
        filtered: list[str] = []
        allowed_ids = set(selected_ids)
        allowed_ids.add(current_topic_id)
        for point in points:
            foreign_reference = False
            for topic_id in self._topics:
                if topic_id in allowed_ids:
                    continue
                if f"`{topic_id}`" in point:
                    foreign_reference = True
                    break
            if not foreign_reference:
                filtered.append(point)
        return filtered or ["-"]
