from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.core.interfaces import KnowledgeBaseSearcher
from src.core.models import FactRecord, RetrievedChunk
from src.retrieval.fact_repository import FactRepository


@dataclass(slots=True)
class _FactDoc:
    fact_id: str
    topic: str
    action_tags: list[str]
    text: str
    aliases: list[str]
    section_tag: str
    tokens: list[str]
    token_counts: Counter[str]
    length: int
    trigrams: set[str]
    priority: int


class FactsHybridSearcher(KnowledgeBaseSearcher):
    """Hybrid retriever over normalized facts (facts.yaml), not KB txt blocks."""

    def __init__(
        self,
        facts_path: Path,
        bm25_weight: float = 0.55,
        dense_weight: float = 0.45,
    ) -> None:
        self._repo = FactRepository(facts_path)
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._docs = self._build_docs(self._repo.list_facts())
        self._avg_doc_len = (
            sum(doc.length for doc in self._docs) / len(self._docs)
            if self._docs
            else 1.0
        )
        self._idf = self._build_idf()

    def search(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        bm25_hits = self.search_bm25(query=query, topic_id=topic_id, top_k=top_k)
        dense_hits = self.search_dense(query=query, topic_id=topic_id, top_k=top_k)
        merged: dict[str, RetrievedChunk] = {}
        for chunk in bm25_hits + dense_hits:
            fact_id = str(chunk.metadata.get("fact_id", ""))
            if not fact_id:
                continue
            if fact_id not in merged or chunk.score > merged[fact_id].score:
                merged[fact_id] = chunk
        out = list(merged.values())
        out.sort(key=lambda item: item.score, reverse=True)
        return out[:top_k]

    def search_bm25(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        results: list[RetrievedChunk] = []
        for doc in self._docs:
            if doc.topic != topic_id and topic_id not in doc.action_tags:
                # keep broad recall but prefer topic-match; skip completely unrelated docs
                continue
            bm25 = self._score_bm25(query_tokens, doc)
            if bm25 <= 0:
                continue
            score = (self._bm25_weight * bm25) + (doc.priority * 0.001) + self._topic_bonus(topic_id, doc)
            results.append(
                RetrievedChunk(
                    text=doc.text,
                    score=score,
                    source="facts.yaml",
                    metadata={
                        "fact_id": doc.fact_id,
                        "entry_index": doc.fact_id,
                        "topic_id": doc.topic,
                        "action_tags": list(doc.action_tags),
                        "search_kind": "bm25",
                        "bm25_score": round(bm25, 6),
                        "section_tag": doc.section_tag,
                    },
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def search_dense(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        query_trigrams = self._trigrams(query)
        if not query_trigrams:
            return []
        results: list[RetrievedChunk] = []
        for doc in self._docs:
            if doc.topic != topic_id and topic_id not in doc.action_tags:
                continue
            dense = self._score_dense(query_trigrams, doc.trigrams)
            if dense <= 0:
                continue
            score = (self._dense_weight * dense) + (doc.priority * 0.001) + self._topic_bonus(topic_id, doc)
            results.append(
                RetrievedChunk(
                    text=doc.text,
                    score=score,
                    source="facts.yaml",
                    metadata={
                        "fact_id": doc.fact_id,
                        "entry_index": doc.fact_id,
                        "topic_id": doc.topic,
                        "action_tags": list(doc.action_tags),
                        "search_kind": "dense",
                        "dense_score": round(dense, 6),
                        "section_tag": doc.section_tag,
                    },
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _build_docs(self, facts: list[FactRecord]) -> list[_FactDoc]:
        docs: list[_FactDoc] = []
        for fact in facts:
            if fact.fact_type != "knowledge":
                continue
            joined_aliases = " ".join(fact.aliases)
            full_text = f"{fact.text}\n{joined_aliases}".strip()
            tokens = self._tokenize(full_text)
            docs.append(
                _FactDoc(
                    fact_id=fact.fact_id,
                    topic=fact.topic,
                    action_tags=list(fact.action_tags),
                    text=fact.text.strip(),
                    aliases=list(fact.aliases),
                    section_tag=str(fact.section_tag or "general"),
                    tokens=tokens,
                    token_counts=Counter(tokens),
                    length=max(len(tokens), 1),
                    trigrams=self._trigrams(full_text),
                    priority=fact.priority,
                )
            )
        return docs

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self._docs)
        if doc_count == 0:
            return {}
        df: Counter[str] = Counter()
        for doc in self._docs:
            for token in set(doc.tokens):
                df[token] += 1
        idf: dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    def _score_bm25(self, query_tokens: list[str], doc: _FactDoc) -> float:
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
        inter = len(query_trigrams & doc_trigrams)
        union = len(query_trigrams | doc_trigrams)
        if union == 0:
            return 0.0
        return inter / union

    @staticmethod
    def _topic_bonus(topic_id: str, doc: _FactDoc) -> float:
        if doc.topic == topic_id:
            return 0.12
        if topic_id in doc.action_tags:
            return 0.08
        return 0.0

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
