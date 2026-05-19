from __future__ import annotations

import re

from src.core.models import RetrievedChunk


class EvidenceReranker:
    """Context-aware reranker over retrieved candidates."""

    def rerank(
        self,
        chunks: list[RetrievedChunk],
        query_variants: list[str],
        action_name: str,
        topic_ids: list[str],
        slots: dict[str, object] | None = None,
        response_plan: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        slots = slots or {}
        boosted: list[RetrievedChunk] = []
        for chunk in chunks:
            extra = 0.0
            text = chunk.text.lower().replace("ё", "е")
            overlap_bonus = self._variant_overlap_bonus(text, query_variants)
            action_bonus = self._action_bonus(text, action_name)
            topic_bonus = self._topic_bonus(text, topic_ids)
            slot_bonus = self._slot_bonus(text, slots)
            plan_bonus = self._plan_bonus(text, response_plan or [])
            extra += overlap_bonus + action_bonus + topic_bonus + slot_bonus + plan_bonus
            reason = (
                f"base={chunk.score:.4f}; overlap={overlap_bonus:.4f}; action={action_bonus:.4f}; "
                f"topic={topic_bonus:.4f}; slots={slot_bonus:.4f}; plan={plan_bonus:.4f}"
            )
            metadata = dict(chunk.metadata)
            metadata["why_selected"] = reason
            boosted.append(
                RetrievedChunk(
                    text=chunk.text,
                    score=chunk.score + extra,
                    source=chunk.source,
                    metadata=metadata,
                )
            )
        boosted.sort(key=lambda item: item.score, reverse=True)
        return boosted[:top_k]

    @staticmethod
    def _variant_overlap_bonus(text: str, variants: list[str]) -> float:
        bonus = 0.0
        for variant in variants:
            tokens = [
                token
                for token in re.findall(r"[\wа-яА-ЯёЁ-]+", variant.lower().replace("ё", "е"))
                if len(token) > 2
            ]
            overlap = sum(1 for token in tokens if token in text)
            bonus += overlap * 0.04
        return bonus

    @staticmethod
    def _action_bonus(text: str, action_name: str) -> float:
        if action_name == "company_services_documents":
            return 0.25 if ("договор" in text or "карточк" in text) else 0.0
        if action_name == "tis_tariffs":
            return 0.25 if "tis" in text or "тис" in text else 0.0
        if action_name == "epc_tariffs":
            return 0.25 if "epc" in text or "епс" in text else 0.0
        if action_name == "brand_availability":
            return 0.2 if "бренд" in text or "каталог" in text else 0.0
        if action_name == "company_services":
            return 0.18 if ("возможност" in text or "тариф" in text or "каталог" in text) else -0.05
        if action_name == "compare_epc_tis":
            if ("epc" in text or "епс" in text) and ("tis" in text or "тис" in text):
                return 0.25
            return -0.08
        return 0.0

    @staticmethod
    def _topic_bonus(text: str, topic_ids: list[str]) -> float:
        bonus = 0.0
        if "company_services_info" in topic_ids and ("могу помочь" in text or "каталог" in text):
            bonus += 0.1
        if "human_operator_request" in topic_ids and "менеджер" in text:
            bonus += 0.1
        return bonus

    @staticmethod
    def _slot_bonus(text: str, slots: dict[str, object]) -> float:
        bonus = 0.0
        brands = slots.get("brands", [])
        if isinstance(brands, list):
            for brand in brands:
                normalized = str(brand).strip().lower()
                if normalized and normalized in text:
                    bonus += 0.08
        return bonus

    @staticmethod
    def _plan_bonus(text: str, response_plan: list[str]) -> float:
        if not response_plan:
            return 0.0
        bonus = 0.0
        for item in response_plan:
            if item == "pricing" and re.search(r"\b(руб|стоим|тариф|цена)\b", text):
                bonus += 0.12
            elif item == "brand_coverage" and re.search(r"\b(бренд|марк|каталог)\b", text):
                bonus += 0.1
            elif item == "catalog_comparison" and re.search(r"\b(отлич|разниц|epc|tis|епс|тис)\b", text):
                bonus += 0.12
            elif item == "checkout_steps" and re.search(r"\b(инн|телефон|период|доступ|qr|счет)\b", text):
                bonus += 0.12
            elif item == "post_payment_access" and re.search(r"\b(после оплаты|доступ|подключ)\b", text):
                bonus += 0.1
            elif item == "multi_user_access" and re.search(r"\b(нескольк|пользоват|доступ)\b", text):
                bonus += 0.1
        return bonus
