from __future__ import annotations

import re
from pathlib import Path

from src.core.models import RetrievedChunk, TopicDefinition
from src.core.interfaces import KnowledgeBaseSearcher


class SimpleTextSearcher(KnowledgeBaseSearcher):
    def __init__(self, kb_path: Path, topics: dict[str, TopicDefinition]) -> None:
        self._kb_path = kb_path
        self._topics = topics
        self._entries = self._load_entries()

    def _load_entries(self) -> list[tuple[str, str]]:
        text = self._kb_path.read_text(encoding="utf-8")
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        entries: list[tuple[str, str]] = []
        current_title = ""
        current_content_parts: list[str] = []

        for block in blocks:
            if re.match(r"^\d+\.", block):
                if current_title or current_content_parts:
                    entries.append((current_title, "\n".join(current_content_parts).strip()))
                current_title = block
                current_content_parts = []
            else:
                current_content_parts.append(block)

        if current_title or current_content_parts:
            entries.append((current_title, "\n".join(current_content_parts).strip()))

        return entries

    def search(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        return self.search_bm25(query=query, topic_id=topic_id, top_k=top_k)

    def search_bm25(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        query_tokens = self._tokenize(query)
        topic_tokens = set(self._topics.get(topic_id, TopicDefinition("", "", "", [])).keywords)

        scored: list[RetrievedChunk] = []
        for index, (title, content) in enumerate(self._entries, start=1):
            text = f"{title}\n{content}"
            text_tokens = self._tokenize(text)
            score = self._score(query_tokens, topic_tokens, text_tokens)
            if score <= 0:
                continue
            scored.append(
                RetrievedChunk(
                    text=text,
                    score=score,
                    source=str(self._kb_path),
                    metadata={"entry_index": str(index), "search_kind": "bm25", "topic_id": topic_id},
                )
            )

        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_k]

    def search_dense(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        # Lightweight fallback dense proxy for compatibility with hybrid pipeline.
        return self.search_bm25(query=query, topic_id=topic_id, top_k=top_k)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[\wа-яА-ЯёЁ-]+", text) if len(token) > 1}

    @staticmethod
    def _score(query_tokens: set[str], topic_tokens: set[str], text_tokens: set[str]) -> float:
        query_overlap = len(query_tokens & text_tokens)
        topic_overlap = len({token.lower() for token in topic_tokens} & text_tokens)
        return query_overlap * 1.0 + topic_overlap * 0.7
