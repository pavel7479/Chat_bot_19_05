from __future__ import annotations

import json
import re
from typing import Any

from src.core.interfaces import LLMProvider
from src.core.models import RetrievalQueryContext


class QueryExtension:
    """Build semantically equivalent query variants without changing intent."""

    _PRODUCT_SYNONYMS = {
        "epc": ["epc", "епс", "ips", "каталог запчастей"],
        "tis": ["tis", "тис", "техинфо", "техническая информация"],
    }

    def __init__(self, llm_provider: LLMProvider | None = None, max_variants: int = 5) -> None:
        self._llm = llm_provider
        self._max_variants = max(3, max_variants)

    def build(
        self,
        user_query: str,
        topic_ids: list[str],
        state_snapshot: dict[str, Any] | None = None,
        turn_analysis: dict[str, Any] | None = None,
    ) -> list[str]:
        context = RetrievalQueryContext(
            trace_id="",
            raw_query=user_query,
            topic_ids=list(topic_ids),
            planned_action=str((turn_analysis or {}).get("planned_action", "")),
            current_focus=str((turn_analysis or {}).get("current_focus", "unknown")),
            slots_snapshot=dict((state_snapshot or {}).get("slots", {}) if isinstance((state_snapshot or {}).get("slots", {}), dict) else {}),
            state_snapshot=dict(state_snapshot or {}),
        )
        return self.build_variants(context=context, turn_analysis=turn_analysis)

    def build_variants(
        self,
        context: RetrievalQueryContext,
        turn_analysis: dict[str, Any] | None = None,
    ) -> list[str]:
        user_query = context.raw_query
        topic_ids = context.topic_ids
        state_snapshot = context.state_snapshot
        base = self._normalize(user_query)
        if not base:
            return []
        variants: list[str] = [base]

        current_focus = str((turn_analysis or {}).get("current_focus", "")).strip().lower()
        slots = ((state_snapshot or {}).get("slots") or {})
        if not isinstance(slots, dict):
            slots = {}

        if current_focus in self._PRODUCT_SYNONYMS:
            variants.extend(self._expand_product(base, current_focus))
        elif "tis_tariffs" in topic_ids:
            variants.extend(self._expand_product(base, "tis"))
        elif "epc_tariffs" in topic_ids:
            variants.extend(self._expand_product(base, "epc"))

        brands = slots.get("brands", [])
        if isinstance(brands, list) and brands:
            joined = " ".join(str(item).strip() for item in brands if str(item).strip())
            if joined:
                variants.append(self._normalize(f"{base} {joined}"))

        if re.search(r"\b(цена|сколько|стоит|тариф)\b", base):
            variants.append(self._normalize(f"{base} стоимость"))
        analysis = turn_analysis or {}
        if bool(analysis.get("catalog_list_request", False)):
            variants.append(self._normalize(f"{base} список брендов каталоги"))
            variants.append(self._normalize(f"{base} какие марки доступны"))
        if bool(analysis.get("feature_comparison", False)):
            variants.append(self._normalize(f"{base} отличие epc tis"))
        if bool(analysis.get("pricing_request", False)):
            variants.append(self._normalize(f"{base} стоимость подписки"))

        variants = self._dedupe(variants)
        llm_variants = self._build_llm_variants(base_query=base, context=context)
        for variant in llm_variants:
            if self.validate_variant(
                base_query=base,
                variant=variant,
                slots=context.slots_snapshot,
            ):
                variants.append(variant)
        return self._dedupe(variants)[: self._max_variants]

    def validate_variant(self, base_query: str, variant: str, slots: dict[str, object] | None = None) -> bool:
        normalized_variant = self._normalize(variant)
        if not normalized_variant:
            return False
        if normalized_variant == self._normalize(base_query):
            return False
        # Guardrail: disallow adding numbers/brands that do not exist in source query/slots.
        base_tokens = set(self._tokenize(base_query))
        variant_tokens = set(self._tokenize(normalized_variant))
        slots = slots or {}
        allowed_brand_tokens: set[str] = set()
        brands = slots.get("brands", [])
        if isinstance(brands, list):
            for item in brands:
                allowed_brand_tokens.update(self._tokenize(str(item)))
        added_tokens = variant_tokens - base_tokens
        risky_tokens = {token for token in added_tokens if token.isdigit() and len(token) >= 2}
        if risky_tokens:
            return False
        # Keep product synonym enrichment but block unrelated entity injection.
        allowed_added = {"epc", "епс", "ips", "tis", "тис", "техинфо", "стоимость", "цена", "тариф", "каталог", "бренд", "марка"}
        for token in added_tokens:
            if token in allowed_added:
                continue
            if token in allowed_brand_tokens:
                continue
            # Single short tokens are typically harmless morphology.
            if len(token) <= 3:
                continue
            return False
        return True

    def _build_llm_variants(self, base_query: str, context: RetrievalQueryContext) -> list[str]:
        if self._llm is None:
            return []
        prompt = (
            "Сделай 2 коротких перефраза запроса без добавления новых фактов, цен, брендов и условий. "
            "Сохрани исходный смысл. Верни JSON: {\"variants\": [\"...\", \"...\"]}.\n"
            f"Запрос: {base_query}\n"
            f"Фокус: {context.current_focus}\n"
            f"Действие: {context.planned_action}\n"
        )
        try:
            raw = self._llm.generate_json(prompt)
            parsed = json.loads(raw)
            values = parsed.get("variants", []) if isinstance(parsed, dict) else []
            if not isinstance(values, list):
                return []
            return [self._normalize(str(v)) for v in values if str(v).strip()]
        except Exception:
            return []

    @classmethod
    def _expand_product(cls, query: str, product_key: str) -> list[str]:
        expanded: list[str] = []
        for synonym in cls._PRODUCT_SYNONYMS.get(product_key, []):
            expanded.append(cls._normalize(f"{query} {synonym}"))
        return expanded

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"[\wа-яА-ЯёЁ-]+", text.lower().replace("ё", "е")) if token]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        result: list[str] = []
        for item in items:
            if item and item not in result:
                result.append(item)
        return result
