from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.core.interfaces import KnowledgeBaseSearcher
from src.core.models import RetrievedChunk, TopicDefinition


@dataclass(slots=True)
class _DocFeatures:
    text: str
    tokens: list[str]
    token_counts: Counter[str]
    length: int
    trigrams: set[str]
    entry_index: int


class HybridSearcher(KnowledgeBaseSearcher):
    """Hybrid retriever: BM25 + lightweight dense similarity."""

    def __init__(
        self,
        kb_path: Path,
        topics: dict[str, TopicDefinition],
        bm25_weight: float = 0.55,
        dense_weight: float = 0.45,
    ) -> None:
        self._kb_path = kb_path
        self._topics = topics
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._docs = self._load_docs()
        self._avg_doc_len = (
            sum(doc.length for doc in self._docs) / len(self._docs)
            if self._docs
            else 1.0
        )
        self._idf = self._build_idf()

    def search(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        bm25 = self.search_bm25(query=query, topic_id=topic_id, top_k=top_k)
        dense = self.search_dense(query=query, topic_id=topic_id, top_k=top_k)
        merged: dict[str, RetrievedChunk] = {}
        for chunk in bm25 + dense:
            key = str(chunk.metadata.get("entry_index", ""))
            if key not in merged or chunk.score > merged[key].score:
                merged[key] = chunk
        result = list(merged.values())
        result.sort(key=lambda item: item.score, reverse=True)
        return result[:top_k]

    def search_bm25(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        if not self._docs:
            return []
        query_tokens = self._tokenize(query)
        topic_tokens = set(self._topics.get(topic_id, TopicDefinition("", "", "", [])).keywords)
        results: list[RetrievedChunk] = []
        for doc in self._docs:
            bm25 = self._score_bm25(query_tokens, doc)
            topic_bonus = self._topic_bonus(topic_tokens, set(doc.tokens))
            score = (self._bm25_weight * bm25) + topic_bonus
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    text=doc.text,
                    score=score,
                    source=str(self._kb_path),
                    metadata={
                        "entry_index": str(doc.entry_index),
                        "search_kind": "bm25",
                        "topic_id": topic_id,
                        "bm25_score": round(bm25, 6),
                    },
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def search_dense(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        if not self._docs:
            return []
        query_trigrams = self._trigrams(query)
        topic_tokens = set(self._topics.get(topic_id, TopicDefinition("", "", "", [])).keywords)
        results: list[RetrievedChunk] = []
        for doc in self._docs:
            dense = self._score_dense(query_trigrams, doc.trigrams)
            topic_bonus = self._topic_bonus(topic_tokens, set(doc.tokens))
            score = (self._dense_weight * dense) + topic_bonus
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    text=doc.text,
                    score=score,
                    source=str(self._kb_path),
                    metadata={
                        "entry_index": str(doc.entry_index),
                        "search_kind": "dense",
                        "topic_id": topic_id,
                        "dense_score": round(dense, 6),
                    },
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self._docs)
        df: Counter[str] = Counter()
        for doc in self._docs:
            for token in set(doc.tokens):
                df[token] += 1
        idf: dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    def _score_bm25(self, query_tokens: list[str], doc: _DocFeatures) -> float:
        if not query_tokens:
            return 0.0
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in query_tokens:
            tf = doc.token_counts.get(token, 0)
            if tf <= 0:
                continue
            idf = self._idf.get(token, 0.0)
            denom = tf + k1 * (1 - b + b * (doc.length / self._avg_doc_len))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-8))
        return score

    @staticmethod
    def _score_dense(query_trigrams: set[str], doc_trigrams: set[str]) -> float:
        if not query_trigrams or not doc_trigrams:
            return 0.0
        inter = len(query_trigrams & doc_trigrams)
        union = len(query_trigrams | doc_trigrams)
        if union == 0:
            return 0.0
        return inter / union

    @staticmethod
    def _topic_bonus(topic_tokens: set[str], doc_tokens: set[str]) -> float:
        if not topic_tokens:
            return 0.0
        overlap = len({item.lower() for item in topic_tokens} & doc_tokens)
        return overlap * 0.1

    def _load_docs(self) -> list[_DocFeatures]:
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

        docs: list[_DocFeatures] = []
        for index, (title, content) in enumerate(entries, start=1):
            combined = f"{title}\n{content}".strip()
            tokens = self._tokenize(combined)
            docs.append(
                _DocFeatures(
                    text=combined,
                    tokens=tokens,
                    token_counts=Counter(tokens),
                    length=max(len(tokens), 1),
                    trigrams=self._trigrams(combined),
                    entry_index=index,
                )
            )
        return docs

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [
            token.lower().replace("ё", "е")
            for token in re.findall(r"[\wа-яА-ЯёЁ-]+", text)
            if len(token) > 1
        ]

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        normalized = re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()
        if len(normalized) < 3:
            return {normalized} if normalized else set()
        return {normalized[i : i + 3] for i in range(len(normalized) - 2)}
