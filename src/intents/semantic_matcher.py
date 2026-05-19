from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.intents.model_semantic_provider import ModelSemanticProvider


@dataclass(slots=True)
class SemanticCandidate:
    intent: str
    score: float
    evidence: str


class SemanticMatcher:
    """Model-first semantic matcher.

    Uses model output as the single source of semantic candidates.
    """

    def __init__(
        self,
        corpus: dict[str, Any],
        model_provider: ModelSemanticProvider | None = None,
        merge_mode: str = "parallel_fallback",
    ) -> None:
        intents_raw = corpus.get("intents", []) if isinstance(corpus, dict) else []
        self._intent_prototypes: dict[str, list[set[str]]] = {}
        self._model_provider = model_provider
        self._last_model_candidates: list[dict[str, object]] = []
        self._last_model_runtime: dict[str, object] = {
            "enabled": bool(model_provider is not None and model_provider.enabled),
            "attempted": False,
            "returned": 0,
        }
        self._last_prompt_trace: dict[str, object] = {}
        mode = str(merge_mode).strip().lower()
        self._merge_mode = mode if mode in {"parallel_fallback", "model_primary"} else "model_primary"
        for item in intents_raw:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip()
            if not intent:
                continue
            vectors: list[set[str]] = []
            for source_text in (
                item.get("label_ru", ""),
                item.get("choose_when", ""),
                item.get("not_choose_when", ""),
            ):
                tokens = self._tokenize(str(source_text))
                if tokens:
                    vectors.append(tokens)
            if not vectors:
                continue
            self._intent_prototypes[intent] = vectors

    def rank(
        self,
        query: str,
        top_k: int = 3,
        context_summary: dict[str, object] | None = None,
    ) -> list[SemanticCandidate]:
        self._last_model_candidates = []
        self._last_model_runtime = {
            "enabled": bool(self._model_provider is not None and self._model_provider.enabled),
            "attempted": False,
            "returned": 0,
        }
        self._last_prompt_trace = {}
        model_scores: dict[str, tuple[float, str]] = {}
        if self._model_provider is not None and self._model_provider.enabled:
            self._last_model_runtime["attempted"] = True
            selected_topics = None
            trace_id = ""
            if isinstance(context_summary, dict):
                raw_selected = context_summary.get("selected_topics", None)
                if isinstance(raw_selected, list):
                    selected_topics = [str(item).strip() for item in raw_selected if str(item).strip()]
                trace_id = str(context_summary.get("trace_id", "")).strip()
            for cand in self._model_provider.score_intents(
                query,
                context_summary=context_summary,
                selected_topics=selected_topics,
                trace_id=trace_id,
            ):
                self._last_model_candidates.append(
                    {"intent": cand.intent, "score": cand.score, "evidence": cand.evidence}
                )
                prev = model_scores.get(cand.intent)
                if prev is None or cand.score > prev[0]:
                    model_scores[cand.intent] = (cand.score, cand.evidence)
            self._last_model_runtime["returned"] = len(self._last_model_candidates)
            self._last_prompt_trace = self._model_provider.get_last_prompt_trace()
        candidates_map: dict[str, SemanticCandidate] = {}
        for intent, (score, evidence) in model_scores.items():
            weighted = max(0.0, min(1.0, score))
            existing = candidates_map.get(intent)
            if existing is None:
                candidates_map[intent] = SemanticCandidate(
                    intent=intent,
                    score=weighted,
                    evidence=f"model:{evidence}" if evidence else "model",
                )
                continue
            merged_score = max(existing.score, weighted)
            merged_evidence = existing.evidence
            if evidence:
                merged_evidence = f"{merged_evidence};model:{evidence}" if merged_evidence else f"model:{evidence}"
            candidates_map[intent] = SemanticCandidate(
                intent=intent,
                score=merged_score,
                evidence=merged_evidence,
            )

        candidates: list[SemanticCandidate] = list(candidates_map.values())
        if not candidates:
            return []
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def get_last_model_candidates(self) -> list[dict[str, object]]:
        return list(self._last_model_candidates)

    def get_last_model_runtime(self) -> dict[str, object]:
        return dict(self._last_model_runtime)

    def get_last_prompt_trace(self) -> dict[str, object]:
        return dict(self._last_prompt_trace)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        raw = text.lower().replace("ё", "е")
        tokens = re.findall(r"[a-zа-я0-9]{2,}", raw)
        return set(tokens)

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left.union(right)
        if not union:
            return 0.0
        return len(left.intersection(right)) / len(union)
