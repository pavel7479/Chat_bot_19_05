from __future__ import annotations

from dataclasses import dataclass

from src.core.models import IntentCandidate


@dataclass(slots=True)
class ResolvedIntent:
    intent: str
    confidence: float
    source: str
    slots: dict[str, object]
    candidates: list[IntentCandidate]
    abstain_reason: str = ""
    fallback_reason: str = ""
    secondary_intent: str = ""


class IntentResolver:
    """Model-first resolver over semantic candidates."""

    def __init__(
        self,
        fast_threshold: float,
        semantic_default_threshold: float,
        semantic_threshold_by_intent: dict[str, float] | None = None,
        resolution_margin: float = 0.05,
        fast_priority_enabled: bool = True,
        semantic_override_margin: float = 0.12,
        model_first_enabled: bool = False,
        abstain_threshold: float = 0.0,
        guardrail_fast_intents: list[str] | None = None,
        semantic_threshold_for_danger_intents: dict[str, float] | None = None,
        signal_intent_bias: dict[str, list[str]] | None = None,
        signal_intent_bias_weights: dict[str, float] | None = None,
        model_evidence_boost: float = 0.0,
        ambiguity_gate_intents: list[str] | None = None,
        ambiguity_gate_margin: float = 0.0,
        ambiguity_gate_margin_by_intent: dict[str, float] | None = None,
        intent_confusion_map: dict[str, list[str]] | None = None,
        dual_secondary_threshold: float = 0.22,
    ) -> None:
        self._fast_threshold = max(0.0, min(1.0, float(fast_threshold)))
        self._semantic_default_threshold = max(0.0, min(1.0, float(semantic_default_threshold)))
        self._semantic_threshold_by_intent = dict(semantic_threshold_by_intent or {})
        self._resolution_margin = max(0.0, min(1.0, float(resolution_margin)))
        self._fast_priority_enabled = bool(fast_priority_enabled)
        self._semantic_override_margin = max(0.0, min(1.0, float(semantic_override_margin)))
        self._model_first_enabled = bool(model_first_enabled)
        self._abstain_threshold = max(0.0, min(1.0, float(abstain_threshold)))
        self._guardrail_fast_intents = {str(item).strip() for item in (guardrail_fast_intents or []) if str(item).strip()}
        self._semantic_threshold_for_danger_intents = {
            str(k).strip(): max(0.0, min(1.0, float(v)))
            for k, v in (semantic_threshold_for_danger_intents or {}).items()
            if str(k).strip()
        }
        self._signal_intent_bias = {
            str(k).strip(): [str(x).strip() for x in vals if str(x).strip()]
            for k, vals in (signal_intent_bias or {}).items()
            if str(k).strip() and isinstance(vals, list)
        }
        self._signal_intent_bias_weights = {
            str(k).strip(): max(0.0, min(0.5, float(v)))
            for k, v in (signal_intent_bias_weights or {}).items()
            if str(k).strip()
        }
        self._model_evidence_boost = max(0.0, min(0.2, float(model_evidence_boost)))
        self._ambiguity_gate_intents = {str(item).strip() for item in (ambiguity_gate_intents or []) if str(item).strip()}
        self._ambiguity_gate_margin = max(0.0, min(1.0, float(ambiguity_gate_margin)))
        self._ambiguity_gate_margin_by_intent = {
            str(k).strip(): max(0.0, min(1.0, float(v)))
            for k, v in (ambiguity_gate_margin_by_intent or {}).items()
            if str(k).strip()
        }
        self._intent_confusion_map = {
            str(k).strip(): {str(item).strip() for item in vals if str(item).strip()}
            for k, vals in (intent_confusion_map or {}).items()
            if str(k).strip() and isinstance(vals, list)
        }
        self._dual_secondary_threshold = max(0.0, min(1.0, float(dual_secondary_threshold)))

    def resolve(
        self,
        fast_candidate: IntentCandidate | None,
        semantic_candidates: list[IntentCandidate],
        signal_hints: set[str] | None = None,
    ) -> ResolvedIntent:
        if self._should_take_fast_guardrail(fast_candidate=fast_candidate):
            assert fast_candidate is not None
            return ResolvedIntent(
                intent=fast_candidate.intent,
                confidence=max(0.0, min(1.0, fast_candidate.score)),
                source="fast",
                slots={},
                candidates=self._merge_candidates(fast_candidate=fast_candidate, semantic_candidates=semantic_candidates),
            )

        # Strict model-first: use semantic/model candidates as-is.
        merged = sorted(
            [
                IntentCandidate(
                    intent=item.intent,
                    score=max(0.0, min(1.0, item.score)),
                    matched_slots={},
                    evidence=item.evidence,
                )
                for item in semantic_candidates
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        if not merged:
            return ResolvedIntent(
                intent="nonsense_input",
                confidence=0.0,
                source="fallback",
                slots={},
                candidates=[],
                fallback_reason="no_candidates",
            )

        top = merged[0]
        if self._should_abstain_on_ambiguity(merged):
            return ResolvedIntent(
                intent="nonsense_input",
                confidence=top.score,
                source="abstain",
                slots={},
                candidates=merged,
                abstain_reason="ambiguity_gate",
            )
        if top.score < self._abstain_threshold:
            return ResolvedIntent(
                intent="nonsense_input",
                confidence=top.score,
                source="abstain",
                slots={},
                candidates=merged,
                abstain_reason="below_abstain_threshold",
            )
        if top.score < self._intent_threshold(top.intent):
            return ResolvedIntent(
                intent="nonsense_input",
                confidence=top.score,
                source="fallback",
                slots={},
                candidates=merged,
                fallback_reason="below_intent_threshold",
            )

        source = "model_semantic" if self._is_model_backed(top) else "semantic"
        return ResolvedIntent(
            intent=top.intent,
            confidence=top.score,
            source=source,
            slots={},
            candidates=merged,
            secondary_intent=self._resolve_secondary_intent(top=top, merged=merged, signal_hints=signal_hints or set()),
        )

    def _merge_candidates(
        self,
        fast_candidate: IntentCandidate | None,
        semantic_candidates: list[IntentCandidate],
    ) -> list[IntentCandidate]:
        # Backward-compat shim. Not used in strict model-first path.
        _ = fast_candidate
        ranked = sorted(
            [
                IntentCandidate(
                    intent=item.intent,
                    score=max(0.0, min(1.0, item.score)),
                    matched_slots={},
                    evidence=item.evidence,
                )
                for item in semantic_candidates
            ],
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked

    def _intent_threshold(self, intent: str) -> float:
        if intent in self._semantic_threshold_for_danger_intents:
            return self._semantic_threshold_for_danger_intents[intent]
        return self._semantic_threshold_by_intent.get(intent, self._semantic_default_threshold)

    def _resolve_source(
        self,
        top: IntentCandidate,
        merged: list[IntentCandidate],
        fast_candidate: IntentCandidate | None,
    ) -> str:
        _ = merged
        _ = fast_candidate
        if self._is_model_backed(top):
            return "model_semantic"
        return "semantic"

    def _accept_fast_candidate(
        self,
        fast_candidate: IntentCandidate | None,
        semantic_candidates: list[IntentCandidate],
    ) -> bool:
        if self._model_first_enabled:
            return False
        if fast_candidate is None:
            return False
        if fast_candidate.score < self._fast_threshold:
            return False
        if not semantic_candidates:
            return True
        top_semantic = semantic_candidates[0]
        if top_semantic.intent == fast_candidate.intent:
            return True
        return top_semantic.score < (fast_candidate.score + self._semantic_override_margin)

    def _should_take_fast_guardrail(self, fast_candidate: IntentCandidate | None) -> bool:
        if fast_candidate is None:
            return False
        if fast_candidate.score < self._fast_threshold:
            return False
        # In model-first mode, fast is allowed only for strict guardrails.
        return fast_candidate.intent in self._guardrail_fast_intents

    @staticmethod
    def _is_model_backed(candidate: IntentCandidate) -> bool:
        return "model:" in candidate.evidence

    def _apply_signal_bias(self, merged: list[IntentCandidate], signal_hints: set[str]) -> list[IntentCandidate]:
        _ = signal_hints
        return merged

    def _apply_model_evidence_boost(self, merged: list[IntentCandidate]) -> list[IntentCandidate]:
        return merged

    def _should_abstain_on_ambiguity(self, merged: list[IntentCandidate]) -> bool:
        if len(merged) < 2:
            return False
        top = merged[0]
        margin = self._ambiguity_gate_margin_by_intent.get(top.intent, self._ambiguity_gate_margin)
        if margin <= 0.0:
            return False
        if top.intent not in self._ambiguity_gate_intents:
            return False
        second = merged[1]
        if not self._is_confusable_pair(top.intent, second.intent):
            return False
        return (top.score - second.score) < margin

    def _is_confusable_pair(self, top_intent: str, second_intent: str) -> bool:
        if not self._intent_confusion_map:
            return True
        neighbors = self._intent_confusion_map.get(top_intent, set())
        return second_intent in neighbors

    def _resolve_secondary_intent(
        self,
        top: IntentCandidate,
        merged: list[IntentCandidate],
        signal_hints: set[str],
    ) -> str:
        if len(merged) < 2:
            return ""
        second = merged[1]
        if second.score < self._dual_secondary_threshold:
            return ""
        if not self._is_confusable_pair(top.intent, second.intent):
            # Allow dual only for explicit multi-intent turns.
            if "multi_intent_marker" not in signal_hints:
                return ""
        return second.intent
