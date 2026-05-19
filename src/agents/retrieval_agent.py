from __future__ import annotations

from uuid import uuid4

from src.core.models import TopicClassificationResult
from src.core.interfaces import KnowledgeBaseSearcher
from src.core.models import RetrievedChunk, RetrievalQueryContext
from src.core.turn_analysis import extract_turn_analysis
from src.retrieval.query_extension import QueryExtension
from src.retrieval.reranker import EvidenceReranker


class RetrievalAgent:
    """Second agent (part 1): retrieve relevant KB chunks by detected topics."""

    def __init__(
        self,
        searcher: KnowledgeBaseSearcher,
        top_k: int,
        max_context_chars: int,
        query_extension: QueryExtension | None = None,
        reranker: EvidenceReranker | None = None,
    ) -> None:
        self._searcher = searcher
        self._top_k = top_k
        self._max_context_chars = max_context_chars
        self._query_extension = query_extension or QueryExtension()
        self._reranker = reranker or EvidenceReranker()

    def retrieve(
        self,
        user_query: str,
        topic_result: TopicClassificationResult,
    ) -> tuple[list[RetrievedChunk], str, dict[str, object]]:
        turn_analysis = extract_turn_analysis(topic_result)
        slots: dict[str, object] = {}
        state_slots = topic_result.state_snapshot.get("slots", {}) if isinstance(topic_result.state_snapshot, dict) else {}
        if isinstance(state_slots, dict):
            slots.update(state_slots)
        slots.update(turn_analysis.slots)
        query_context = RetrievalQueryContext(
            trace_id=str(topic_result.retrieval_context.get("trace_id", "")) if isinstance(topic_result.retrieval_context, dict) else "",
            raw_query=user_query,
            topic_ids=list(topic_result.topic_ids),
            planned_action=topic_result.planned_action,
            current_focus=topic_result.current_focus,
            slots_snapshot=dict(slots),
            state_snapshot=dict(topic_result.state_snapshot),
        )
        if not query_context.trace_id:
            query_context.trace_id = uuid4().hex
        query_variants = self._query_extension.build_variants(
            context=query_context,
            turn_analysis={
                "current_focus": turn_analysis.current_focus,
                "slots": dict(turn_analysis.slots),
                "pricing_request": turn_analysis.pricing_request,
                "catalog_list_request": turn_analysis.catalog_list_request,
                "feature_comparison": turn_analysis.feature_comparison,
            },
        )
        query_context.query_variants = list(query_variants or [user_query])
        unique: dict[tuple[str, str], RetrievedChunk] = {}
        dropped_candidates: list[dict[str, object]] = []
        raw_hits_bm25: list[dict[str, object]] = []
        raw_hits_dense: list[dict[str, object]] = []
        for topic_id in topic_result.topic_ids:
            for query in query_context.query_variants:
                bm25_chunks = self._search_bm25(
                    query=query,
                    topic_id=topic_id,
                    top_k=self._top_k,
                )
                dense_chunks = self._search_dense(
                    query=query,
                    topic_id=topic_id,
                    top_k=self._top_k,
                )
                raw_hits_bm25.append(
                    {
                        "query": query,
                        "topic_id": topic_id,
                        "hits": [
                            {
                                "score": round(chunk.score, 4),
                                "source": chunk.source,
                                "entry_index": chunk.metadata.get("entry_index", ""),
                                "fact_id": chunk.metadata.get("fact_id", ""),
                                "search_kind": chunk.metadata.get("search_kind", "bm25"),
                                "section_tag": chunk.metadata.get("section_tag", "general"),
                            }
                            for chunk in bm25_chunks
                        ],
                    }
                )
                raw_hits_dense.append(
                    {
                        "query": query,
                        "topic_id": topic_id,
                        "hits": [
                            {
                                "score": round(chunk.score, 4),
                                "source": chunk.source,
                                "entry_index": chunk.metadata.get("entry_index", ""),
                                "fact_id": chunk.metadata.get("fact_id", ""),
                                "search_kind": chunk.metadata.get("search_kind", "dense"),
                                "section_tag": chunk.metadata.get("section_tag", "general"),
                            }
                            for chunk in dense_chunks
                        ],
                    }
                )
                for chunk in bm25_chunks + dense_chunks:
                    key = (chunk.source, chunk.metadata.get("entry_index", ""))
                    if key not in unique:
                        unique[key] = chunk
                        continue
                    prev = unique[key]
                    if chunk.score > prev.score:
                        dropped_candidates.append(
                            {
                                "entry_index": prev.metadata.get("entry_index", ""),
                                "fact_id": prev.metadata.get("fact_id", ""),
                                "score": round(prev.score, 4),
                                "source": prev.source,
                                "reason": "duplicate_lower_score_replaced",
                            }
                        )
                        unique[key] = chunk
                    else:
                        dropped_candidates.append(
                            {
                                "entry_index": chunk.metadata.get("entry_index", ""),
                                "fact_id": chunk.metadata.get("fact_id", ""),
                                "score": round(chunk.score, 4),
                                "source": chunk.source,
                                "reason": "duplicate_lower_score_discarded",
                            }
                        )

        merged_candidates = list(unique.values())
        chunks = self._reranker.rerank(
            chunks=merged_candidates,
            query_variants=query_context.query_variants,
            action_name=topic_result.planned_action,
            topic_ids=topic_result.topic_ids,
            slots=slots,
            response_plan=topic_result.response_plan,
            top_k=self._top_k,
        )
        chunks = chunks[: self._top_k]
        selected_keys = {(chunk.source, chunk.metadata.get("entry_index", "")) for chunk in chunks}
        dropped_topk = []
        for chunk in merged_candidates:
            key = (chunk.source, chunk.metadata.get("entry_index", ""))
            if key in selected_keys:
                continue
            dropped_topk.append(
                {
                    "entry_index": chunk.metadata.get("entry_index", ""),
                    "score": round(chunk.score, 4),
                    "source": chunk.source,
                    "reason": "not_in_top_k_after_rerank",
                }
            )

        context = "\n\n".join(chunk.text for chunk in chunks)
        trace = {
            "trace_id": query_context.trace_id,
            "query_received": {
                "query": user_query,
                "topic_ids": topic_result.topic_ids,
                "planned_action": topic_result.planned_action,
                "current_focus": topic_result.current_focus,
                "slots": slots,
            },
            "query_extended": {
                "variants": list(query_context.query_variants),
            },
            "bm25_hits": raw_hits_bm25,
            "dense_hits": raw_hits_dense,
            "merged_candidates": [
                {
                    "score": round(chunk.score, 4),
                    "source": chunk.source,
                    "entry_index": chunk.metadata.get("entry_index", ""),
                    "fact_id": chunk.metadata.get("fact_id", ""),
                    "search_kind": chunk.metadata.get("search_kind", ""),
                    "section_tag": chunk.metadata.get("section_tag", "general"),
                }
                for chunk in merged_candidates
            ],
            "reranked_topk": [
                {
                    "score": round(chunk.score, 4),
                    "source": chunk.source,
                    "entry_index": chunk.metadata.get("entry_index", ""),
                    "fact_id": chunk.metadata.get("fact_id", ""),
                    "why_selected": str(chunk.metadata.get("why_selected", "")),
                    "section_tag": chunk.metadata.get("section_tag", "general"),
                }
                for chunk in chunks
            ],
            "dropped_candidates": dropped_candidates + dropped_topk,
        }
        return chunks, context[: self._max_context_chars], trace

    def _search_bm25(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        method = getattr(self._searcher, "search_bm25", None)
        if callable(method):
            return method(query=query, topic_id=topic_id, top_k=top_k)
        return self._searcher.search(query=query, topic_id=topic_id, top_k=top_k)

    def _search_dense(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        method = getattr(self._searcher, "search_dense", None)
        if callable(method):
            return method(query=query, topic_id=topic_id, top_k=top_k)
        return self._searcher.search(query=query, topic_id=topic_id, top_k=top_k)
