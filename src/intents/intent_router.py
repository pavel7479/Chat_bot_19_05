from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import yaml

from src.core.models import IntentCandidate, RouteDecision
from src.intents.fast_matcher import FastMatcher
from src.intents.fsm_orchestrator import FSMOrchestrator
from src.intents.intent_resolver import IntentResolver
from src.intents.model_semantic_provider import ModelSemanticProvider
from src.intents.semantic_matcher import SemanticMatcher


class IntentRouter:
    """Primary routing: FastMatcher + SemanticMatcher -> IntentResolver -> FSM."""

    def __init__(
        self,
        fast_rules_path: Path,
        semantic_intents_path: Path,
        fsm_path: Path,
        thresholds_path: Path,
        model_provider: ModelSemanticProvider | None = None,
        merge_mode: str = "parallel_fallback",
    ) -> None:
        fast_rules = self._load_yaml(fast_rules_path)
        semantic_corpus = self._load_yaml(semantic_intents_path)
        fsm_spec = self._load_yaml(fsm_path)
        thresholds = self._load_yaml(thresholds_path)
        self._fast = FastMatcher(fast_rules)
        self._semantic = SemanticMatcher(
            semantic_corpus,
            model_provider=model_provider,
            merge_mode=merge_mode,
        )
        self._intent_labels_ru = self._load_intent_labels_ru(semantic_corpus)
        self._fsm = FSMOrchestrator(fsm_spec)
        self._fast_threshold = float(thresholds.get("fast_confidence", 0.95))
        self._semantic_threshold = float(thresholds.get("semantic_confidence", 0.35))
        self._resolution_margin = float(thresholds.get("intent_resolution_margin", 0.05))
        self._fast_priority_enabled = bool(thresholds.get("fast_priority_enabled", True))
        self._semantic_override_margin = float(thresholds.get("semantic_override_margin", 0.12))
        self._model_first_enabled = bool(thresholds.get("model_first_enabled", False))
        self._abstain_threshold = float(thresholds.get("abstain_threshold", 0.0))
        self._guardrail_fast_intents: list[str] = []
        raw_guardrail = thresholds.get("guardrail_fast_intents", [])
        if isinstance(raw_guardrail, list):
            self._guardrail_fast_intents = [str(item).strip() for item in raw_guardrail if str(item).strip()]
        self._semantic_threshold_by_intent: dict[str, float] = {}
        raw_semantic_by_intent = thresholds.get("semantic_confidence_by_intent", {})
        if isinstance(raw_semantic_by_intent, dict):
            for intent, value in raw_semantic_by_intent.items():
                intent_name = str(intent).strip()
                if not intent_name:
                    continue
                try:
                    threshold = float(value)
                except (TypeError, ValueError):
                    continue
                self._semantic_threshold_by_intent[intent_name] = max(0.0, min(1.0, threshold))
        self._semantic_threshold_for_danger_intents: dict[str, float] = {}
        raw_danger = thresholds.get("semantic_confidence_for_danger_intents", {})
        if isinstance(raw_danger, dict):
            for intent, value in raw_danger.items():
                try:
                    self._semantic_threshold_for_danger_intents[str(intent).strip()] = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
        self._signal_intent_bias: dict[str, list[str]] = {}
        raw_signal_bias = thresholds.get("signal_intent_bias", {})
        if isinstance(raw_signal_bias, dict):
            for signal, intents in raw_signal_bias.items():
                if isinstance(intents, list):
                    self._signal_intent_bias[str(signal).strip()] = [str(item).strip() for item in intents if str(item).strip()]
        self._signal_intent_bias_weights: dict[str, float] = {}
        raw_signal_bias_weights = thresholds.get("signal_intent_bias_weights", {})
        if isinstance(raw_signal_bias_weights, dict):
            for signal, weight in raw_signal_bias_weights.items():
                try:
                    self._signal_intent_bias_weights[str(signal).strip()] = max(0.0, min(0.5, float(weight)))
                except (TypeError, ValueError):
                    continue
        self._model_evidence_boost = float(thresholds.get("model_evidence_boost", 0.0))
        self._dual_secondary_threshold = float(thresholds.get("dual_secondary_threshold", 0.22))
        raw_ambiguity_intents = thresholds.get("ambiguity_gate_intents", [])
        self._ambiguity_gate_intents: list[str] = []
        if isinstance(raw_ambiguity_intents, list):
            self._ambiguity_gate_intents = [str(item).strip() for item in raw_ambiguity_intents if str(item).strip()]
        self._ambiguity_gate_margin = float(thresholds.get("ambiguity_gate_margin", 0.0))
        self._ambiguity_gate_margin_by_intent: dict[str, float] = {}
        raw_ambiguity_margin_by_intent = thresholds.get("ambiguity_gate_margin_by_intent", {})
        if isinstance(raw_ambiguity_margin_by_intent, dict):
            for intent, margin in raw_ambiguity_margin_by_intent.items():
                intent_name = str(intent).strip()
                if not intent_name:
                    continue
                try:
                    self._ambiguity_gate_margin_by_intent[intent_name] = max(0.0, min(1.0, float(margin)))
                except (TypeError, ValueError):
                    continue
        self._intent_confusion_map = self._load_intent_confusion_map(semantic_corpus)
        self._resolver = IntentResolver(
            fast_threshold=self._fast_threshold,
            semantic_default_threshold=self._semantic_threshold,
            semantic_threshold_by_intent=self._semantic_threshold_by_intent,
            resolution_margin=self._resolution_margin,
            fast_priority_enabled=self._fast_priority_enabled,
            semantic_override_margin=self._semantic_override_margin,
            model_first_enabled=self._model_first_enabled,
            abstain_threshold=self._abstain_threshold,
            guardrail_fast_intents=self._guardrail_fast_intents,
            semantic_threshold_for_danger_intents=self._semantic_threshold_for_danger_intents,
            signal_intent_bias=self._signal_intent_bias,
            signal_intent_bias_weights=self._signal_intent_bias_weights,
            model_evidence_boost=self._model_evidence_boost,
            ambiguity_gate_intents=self._ambiguity_gate_intents,
            ambiguity_gate_margin=self._ambiguity_gate_margin,
            ambiguity_gate_margin_by_intent=self._ambiguity_gate_margin_by_intent,
            intent_confusion_map=self._intent_confusion_map,
            dual_secondary_threshold=self._dual_secondary_threshold,
        )
        self._last_model_candidates: list[dict[str, object]] = []
        self._last_model_runtime: dict[str, object] = {"enabled": False, "attempted": False, "returned": 0}
        self._last_prompt_trace: dict[str, object] = {}

    def route(
        self,
        query: str,
        state: str = "discovery",
        context_summary: dict[str, object] | None = None,
    ) -> RouteDecision:
        fast = self._fast.match(query)
        fast_candidate: IntentCandidate | None = None
        if fast is not None:
            fast_candidate = IntentCandidate(
                intent=fast.intent,
                score=max(0.0, min(1.0, fast.confidence)),
                matched_slots=dict(fast.slots),
                evidence=fast.evidence,
            )

        candidates_raw = self._semantic.rank(query, context_summary=context_summary)
        self._last_model_candidates = self._semantic.get_last_model_candidates()
        self._last_model_runtime = self._semantic.get_last_model_runtime()
        self._last_prompt_trace = self._semantic.get_last_prompt_trace()
        candidates = [
            IntentCandidate(
                intent=item.intent,
                score=item.score,
                matched_slots={},
                evidence=item.evidence,
            )
            for item in candidates_raw
        ]

        resolved = self._resolver.resolve(
            fast_candidate=fast_candidate,
            semantic_candidates=candidates,
            signal_hints=self._derive_signal_hints(query),
        )
        fsm_intent = resolved.intent if resolved.intent != "nonsense_input" else "clarify_request"
        fsm = self._fsm.transition(state=state, intent=fsm_intent)
        return RouteDecision(
            intent=resolved.intent,
            slots={},
            confidence=resolved.confidence,
            source=resolved.source,
            candidates=resolved.candidates,
            planned_action=fsm.planned_action,
            next_state=fsm.state,
            abstain_reason=resolved.abstain_reason,
            fallback_reason=resolved.fallback_reason,
        )

    def get_last_model_candidates(self) -> list[dict[str, object]]:
        return list(self._last_model_candidates)

    def get_last_model_runtime(self) -> dict[str, object]:
        return dict(self._last_model_runtime)

    def get_last_prompt_trace(self) -> dict[str, object]:
        return dict(self._last_prompt_trace)

    def get_intent_labels_ru(self) -> dict[str, str]:
        return dict(self._intent_labels_ru)

    @staticmethod
    def _derive_signal_hints(query: str) -> set[str]:
        low = query.lower().replace("ё", "е").strip()
        hints: set[str] = set()
        if re.search(r"(\bи\b|\bа также\b|\bплюс\b|,)", low):
            hints.add("multi_intent_marker")
        if re.search(r"(точн|актуальн|устаревш|ошибк\w*\s+в\s+данных)", low):
            hints.add("data_accuracy")
        if re.search(r"(после\s+оплат\w*.*нет.*доступ|не\s+работает\s+вход)", low):
            hints.add("post_payment_no_access")
        if re.search(r"(после\s+оплат\w*.*сразу|когда\s+после\s+оплат\w*\s+доступ)", low):
            hints.add("post_payment_access_timing")
        if re.search(r"^\W*$|^[a-zа-я]+\d+$", low):
            hints.add("nonsense")
        if re.search(r"\b(туп(ой|ая|о|ые|и(шь)?)?|мошенн\w*|идиот\w*|дурак\w*|бред)\b", low):
            hints.add("abuse")
        return hints

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _load_intent_labels_ru(semantic_corpus: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        intents = semantic_corpus.get("intents", []) if isinstance(semantic_corpus, dict) else []
        for item in intents:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip()
            label_ru = str(item.get("label_ru", "")).strip()
            if intent and label_ru:
                result[intent] = label_ru
        return result

    @staticmethod
    def _load_intent_confusion_map(semantic_corpus: dict[str, Any]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        intents = semantic_corpus.get("intents", []) if isinstance(semantic_corpus, dict) else []
        for item in intents:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip()
            confused_with = item.get("confused_with", [])
            if not intent or not isinstance(confused_with, list):
                continue
            neighbors = [str(x).strip() for x in confused_with if str(x).strip()]
            if neighbors:
                result[intent] = neighbors
        return result
