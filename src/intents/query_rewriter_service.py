from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class RewriteResult:
    rewritten_query: str
    reason: str
    changed: bool


class QueryRewriterService:
    """Context-aware rewrite for intent classification.

    The service never adds new facts; it only makes short replies explicit.
    """

    _SHORT_REPLY_RE = re.compile(
        r"^\s*(да|нет|ага|неа|ок|окей|yes|no|nope|являюсь|не являюсь)\s*[.!?]?\s*$",
        flags=re.IGNORECASE,
    )
    _HAS_BRAND_AVAILABILITY_RE = re.compile(
        r"(есть|имеется|доступен|наличие|каталог).{0,24}\?",
        flags=re.IGNORECASE,
    )
    _BRAND_FIXES: tuple[tuple[str, str], ...] = (
        ("тайота", "Toyota"),
        ("тоета", "Toyota"),
        ("лекссс", "Lexus"),
        ("лексус", "Lexus"),
        ("мерседес", "Mercedes"),
        ("мерс", "Mercedes"),
        ("фольцваген", "Volkswagen"),
        ("фольксваген", "Volkswagen"),
        ("вольво", "Volvo"),
        ("ауди", "Audi"),
        ("бмв", "BMW"),
        ("порш", "Porsche"),
        ("porshe", "Porsche"),
        ("porsch", "Porsche"),
        ("renauldt", "Renault"),
        ("pegeot", "Peugeot"),
        ("hyndai", "Hyundai"),
        ("киаа", "KIA"),
    )

    def rewrite(self, user_query: str, context_summary: dict[str, object] | None = None) -> RewriteResult:
        query = (user_query or "").strip()
        if not query:
            return RewriteResult(rewritten_query=query, reason="empty_query", changed=False)

        context = context_summary or {}

        short_brand_rewrite = self._rewrite_short_brand_reply(query=query, context_summary=context)
        if short_brand_rewrite.changed:
            return short_brand_rewrite

        brand_rewrite = self._rewrite_brand_availability_query(query)
        if brand_rewrite.changed:
            return brand_rewrite

        if not self._SHORT_REPLY_RE.match(query):
            return RewriteResult(rewritten_query=query, reason="no_rewrite_needed", changed=False)

        last_bot_question = str(context.get("last_bot_question", "")).strip()
        if not last_bot_question:
            return RewriteResult(rewritten_query=query, reason="short_reply_no_context", changed=False)

        rewritten = (
            "Клиент дал короткий ответ на предыдущий вопрос бота. "
            f"Предыдущий вопрос: {last_bot_question}. Ответ клиента: {query}."
        )
        return RewriteResult(rewritten_query=rewritten, reason="short_reply_with_context", changed=True)

    def _rewrite_brand_availability_query(self, query: str) -> RewriteResult:
        low = query.lower().replace("ё", "е")
        if not self._HAS_BRAND_AVAILABILITY_RE.search(low):
            return RewriteResult(rewritten_query=query, reason="no_rewrite_needed", changed=False)
        normalized = query
        canonical_hits: list[str] = []
        for raw, canonical in self._BRAND_FIXES:
            pattern = re.compile(rf"\b{re.escape(raw)}\b", flags=re.IGNORECASE)
            if pattern.search(normalized):
                normalized = pattern.sub(canonical, normalized)
                if canonical not in canonical_hits:
                    canonical_hits.append(canonical)
        if not canonical_hits:
            return RewriteResult(rewritten_query=query, reason="no_rewrite_needed", changed=False)
        if len(canonical_hits) == 1:
            rewritten = f"Есть ли у вас бренд {canonical_hits[0]}?"
        else:
            rewritten = f"Есть ли у вас бренды {', '.join(canonical_hits)}?"
        return RewriteResult(
            rewritten_query=rewritten,
            reason="brand_typo_normalized_with_context",
            changed=True,
        )

    def _rewrite_short_brand_reply(self, query: str, context_summary: dict[str, object]) -> RewriteResult:
        """Rewrites short brand-only replies using context from prior turn.

        Keeps behavior general by gating on short input + brand-related previous bot turn.
        """
        tokens = re.findall(r"[A-Za-zА-Яа-я0-9\-]+", query)
        if not (1 <= len(tokens) <= 4):
            return RewriteResult(rewritten_query=query, reason="no_rewrite_needed", changed=False)

        normalized = query
        canonical_hits: list[str] = []
        for raw, canonical in self._BRAND_FIXES:
            pattern = re.compile(rf"\b{re.escape(raw)}\b", flags=re.IGNORECASE)
            if pattern.search(normalized):
                normalized = pattern.sub(canonical, normalized)
                if canonical not in canonical_hits:
                    canonical_hits.append(canonical)

        if not canonical_hits:
            return RewriteResult(rewritten_query=query, reason="no_rewrite_needed", changed=False)

        last_bot_question = str(context_summary.get("last_bot_question", "")).lower().replace("ё", "е")
        previous_primary_topic = str(context_summary.get("previous_primary_topic", "")).strip().lower()
        is_brand_context = (
            "бренд" in last_bot_question
            or "марк" in last_bot_question
            or "каталог" in last_bot_question
            or previous_primary_topic in {"brand_list_request", "specific_brand_check"}
        )
        if not is_brand_context:
            return RewriteResult(rewritten_query=query, reason="no_rewrite_needed", changed=False)

        if len(canonical_hits) == 1:
            rewritten = f"Интересующая марка: {canonical_hits[0]}."
        else:
            rewritten = f"Интересующие марки: {', '.join(canonical_hits)}."
        return RewriteResult(
            rewritten_query=rewritten,
            reason="short_brand_reply_with_context",
            changed=True,
        )
