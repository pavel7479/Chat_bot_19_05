from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class TopicSelectionResult:
    selected_intents: list[str]
    selected_reasons: dict[str, str]
    dropped_reasons: dict[str, str]


class IntentPromptTopicSelector:
    """Selects a compact intent subset for model prompt."""

    def __init__(
        self,
        intents: list[str],
        intent_labels_ru: dict[str, str] | None = None,
        intent_hints: dict[str, dict[str, object]] | None = None,
        top_n: int = 12,
        min_score: float = 0.0,
    ) -> None:
        self._intents = [item for item in intents if item]
        self._labels_ru = dict(intent_labels_ru or {})
        self._hints = dict(intent_hints or {})
        self._top_n = max(6, int(top_n))
        self._min_score = max(0.0, min(1.0, float(min_score)))

    def select(self, user_query: str, context_summary: dict[str, object] | None = None) -> TopicSelectionResult:
        ctx = context_summary or {}
        query_tokens = self._tokenize(user_query)
        context_tokens = self._tokenize(" ".join(self._context_texts(ctx)))
        merged_tokens = query_tokens.union(context_tokens)
        previous_primary_topic = str(ctx.get("previous_primary_topic", "")).strip()

        scored: list[tuple[float, str, str]] = []
        for intent in self._intents:
            reason_parts: list[str] = []
            score = 0.0
            label_tokens = self._tokenize(self._labels_ru.get(intent, ""))
            choose_tokens = self._tokenize(str(self._hints.get(intent, {}).get("choose_when", "")))
            avoid_tokens = self._tokenize(str(self._hints.get(intent, {}).get("not_choose_when", "")))
            proto_tokens = set()
            if not choose_tokens and label_tokens:
                proto_tokens = label_tokens

            overlap = len(merged_tokens.intersection(label_tokens.union(choose_tokens).union(proto_tokens)))
            if overlap > 0:
                score += min(0.8, overlap * 0.12)
                reason_parts.append(f"token_overlap={overlap}")
            # Semantic-like score over descriptive tokens, not only exact overlaps.
            semantic_base = label_tokens.union(choose_tokens).union(proto_tokens)
            semantic_score = self._jaccard(merged_tokens, semantic_base)
            if semantic_score > 0:
                score += min(0.6, semantic_score * 0.5)
                reason_parts.append(f"semantic={semantic_score:.3f}")
            avoid_overlap = len(merged_tokens.intersection(avoid_tokens))
            if avoid_overlap > 0:
                score -= min(0.35, avoid_overlap * 0.08)
                reason_parts.append(f"not_choose_overlap={avoid_overlap}")
            if previous_primary_topic and intent == previous_primary_topic:
                score += 0.28
                reason_parts.append("previous_primary_topic")
            # Continuation turns should keep prior dialog topic available in shortlist.
            if self._is_continuation_turn(ctx) and previous_primary_topic and intent == previous_primary_topic:
                score += 0.16
                reason_parts.append("continuation_bias")

            if intent in {"nonsense_input", "out_of_scope_request"}:
                score += 0.05
                reason_parts.append("service_intent_always_available")

            scored.append((score, intent, ";".join(reason_parts) if reason_parts else "no_overlap"))

        scored.sort(key=lambda item: item[0], reverse=True)
        pre_selected = [item for item in scored if item[0] >= self._min_score][: self._top_n]
        if not pre_selected:
            # Keep prompt focused: when nothing is relevant, pass only fallback topic.
            if "nonsense_input" in self._intents:
                pre_selected = [(0.05, "nonsense_input", "fallback_only_no_relevance")]
            elif "out_of_scope_request" in self._intents:
                pre_selected = [(0.05, "out_of_scope_request", "fallback_only_no_relevance")]
            else:
                pre_selected = scored[:1]
        selected = {intent for _, intent, _ in pre_selected}

        # Add confused_with neighbors for disambiguation.
        for _, intent, _ in pre_selected:
            for neighbor in self._confused_with(intent):
                if neighbor in self._intents:
                    selected.add(neighbor)
        # Explicitly keep previous topic and neighbors for contextual continuity.
        if previous_primary_topic in self._intents:
            selected.add(previous_primary_topic)
            for neighbor in self._confused_with(previous_primary_topic):
                if neighbor in self._intents:
                    selected.add(neighbor)
        if "nonsense_input" in self._intents:
            selected.add("nonsense_input")
        if "out_of_scope_request" in self._intents:
            selected.add("out_of_scope_request")

        selected_reasons: dict[str, str] = {}
        for score, intent, reason in scored:
            if intent in selected:
                selected_reasons[intent] = f"score={score:.3f};{reason}"

        dropped_reasons: dict[str, str] = {}
        for score, intent, reason in scored:
            if intent not in selected:
                dropped_reasons[intent] = f"score={score:.3f};{reason}"

        ordered_selected = [intent for _, intent, _ in scored if intent in selected]
        return TopicSelectionResult(
            selected_intents=ordered_selected,
            selected_reasons=selected_reasons,
            dropped_reasons=dropped_reasons,
        )

    def _confused_with(self, intent: str) -> list[str]:
        value = self._hints.get(intent, {}).get("confused_with", [])
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _context_texts(context_summary: dict[str, object]) -> list[str]:
        texts: list[str] = []
        for key in ("last_bot_question", "history_tail", "previous_primary_topic"):
            raw = context_summary.get(key, "")
            if isinstance(raw, list):
                texts.extend(str(item) for item in raw)
            else:
                texts.append(str(raw))
        return texts

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        raw = (text or "").lower().replace("ё", "е")
        tokens = re.findall(r"[a-zа-я0-9]{2,}", raw)
        stopwords = {
            "и",
            "а",
            "но",
            "или",
            "по",
            "про",
            "на",
            "в",
            "с",
            "к",
            "что",
            "как",
            "ли",
            "это",
            "так",
            "же",
            "для",
            "из",
            "о",
            "об",
            "от",
        }
        return {token for token in tokens if token not in stopwords}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left.union(right)
        if not union:
            return 0.0
        return len(left.intersection(right)) / len(union)

    @staticmethod
    def _is_continuation_turn(context_summary: dict[str, object]) -> bool:
        query = str(context_summary.get("user_query", "")).lower()
        history_tail = context_summary.get("history_tail", [])
        history_len = len(history_tail) if isinstance(history_tail, list) else 0
        short = len(query.split()) <= 4
        followup_markers = ("а ", "и ", "тогда", "ок", "понял", "ага", "да", "нет")
        marker = any(query.startswith(m) for m in followup_markers)
        return history_len > 0 and (short or marker)
