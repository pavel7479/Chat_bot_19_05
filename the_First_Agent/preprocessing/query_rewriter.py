from __future__ import annotations

import re

from src.core.models import SessionState
from the_First_Agent.context.prompt_context import PromptContext


class QueryRewriteService:
    def __init__(self, brand_aliases: list[str]) -> None:
        self._brand_aliases = [alias for alias in brand_aliases if alias]

    def rewrite(
        self,
        context: PromptContext,
        session_state: SessionState | None,
        normalized_query: str,
    ) -> dict[str, object]:
        original = normalized_query
        query = normalized_query.lower()
        last_focus = session_state.last_focus_topic if session_state else "out_of_scope_request"
        last_brand, brand_resolution_trace = self._resolve_active_brand_context(context, session_state, query, last_focus)

        rewritten = original
        mode = "direct"
        reason = "Rewrite disabled for the main pipeline."

        if self._has_brand_anaphora(query) and last_brand:
            rewritten = self._replace_anaphora_with_brand(original, last_brand)
            mode = "anaphora_followup"
            reason = "Safe anaphora replacement using the latest brand context."

        return {
            "mode": mode,
            "original_query": original,
            "rewritten_query": rewritten,
            "changed": rewritten != original,
            "reason": reason,
            "brand_resolution_trace": brand_resolution_trace,
        }

    def _resolve_active_brand_context(
        self,
        context: PromptContext,
        session_state: SessionState | None,
        query: str,
        last_focus: str,
    ) -> tuple[str, dict[str, str]]:
        del last_focus
        query_brand = self._find_last_brand_in_text(query)
        if query_brand:
            return query_brand, {"source": "query", "brand": query_brand}

        session_brand = str(session_state.last_mentioned_brand).strip() if session_state else ""
        history_brand = self._find_last_brand_in_text(context.history_text)
        if self._has_brand_anaphora(query):
            brand = session_brand or history_brand
            if brand:
                return brand, {"source": "anaphora", "brand": brand}

        return "", {"source": "none", "brand": ""}

    @staticmethod
    def _replace_anaphora_with_brand(query: str, brand: str) -> str:
        replaced = re.sub(r"\b(на него|по нему|для него)\b", f"для {brand}", query, flags=re.IGNORECASE)
        return re.sub(r"\b(него|неё|нее|них|нему|ней|этот|эта|эту|эти)\b", brand, replaced, flags=re.IGNORECASE)

    @staticmethod
    def _has_brand_anaphora(query: str) -> bool:
        return bool(re.search(r"\b(на него|по нему|для него|него|неё|нее|них|нему|ней|этот|эта|эту|эти)\b", query))

    def _find_last_brand_in_text(self, text: str) -> str:
        normalized = str(text or "").lower()
        last_match = ""
        for alias in self._brand_aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                last_match = alias
        return last_match.title() if last_match else ""
