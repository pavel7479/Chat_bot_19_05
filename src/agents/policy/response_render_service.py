from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RenderContext:
    action_name: str
    history_text: str
    user_query: str
    topic_ids: list[str]
    state: Any


class TemplateFallbackService:
    def pick(self, action_name: str, templates: dict[str, list[str]], history_text: str) -> str | None:
        variants = templates.get(action_name, [])
        if not variants:
            return None
        recent_answers = self._recent_assistant_answers(history_text, limit=2)
        recent_norm = {self._normalize_answer(item) for item in recent_answers}
        for variant in variants:
            if self._normalize_answer(variant) not in recent_norm:
                return variant
        return variants[0]

    @staticmethod
    def _recent_assistant_answers(history_text: str, limit: int = 2) -> list[str]:
        lines = [line.strip() for line in history_text.splitlines() if line.strip()]
        answers: list[str] = []
        for line in reversed(lines):
            if line.lower().startswith("assistant:"):
                text = line.split(":", 1)[1].strip()
                if text:
                    answers.append(text)
            if len(answers) >= limit:
                break
        return answers

    @staticmethod
    def _normalize_answer(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


class PricingStrategy:
    def __init__(self, brand_resolver: Any, tis_prices: Any) -> None:
        self._brand_resolver = brand_resolver
        self._tis_prices = tis_prices

    def render(self, context: RenderContext) -> str | None:
        if context.action_name != "tis_tariffs":
            return None
        return None


class BrandAvailabilityStrategy:
    def __init__(self, brand_resolver: Any, tis_prices: Any) -> None:
        self._brand_resolver = brand_resolver
        self._tis_prices = tis_prices

    def render(self, context: RenderContext) -> str | None:
        if context.action_name != "brand_availability":
            return None
        return None


class ComparisonStrategy:
    def render(self, context: RenderContext) -> str | None:
        if context.action_name != "compare_epc_tis":
            return None
        return None


class CheckoutStrategy:
    def render(self, context: RenderContext) -> str | None:
        if context.action_name not in {"request_requisites", "ask_legal_status"}:
            return None
        return None


class OutOfScopeStrategy:
    def render(self, context: RenderContext) -> str | None:
        if context.action_name != "out_of_scope_response":
            return None
        return None


class ResponseRenderService:
    def __init__(
        self,
        fact_repository: Any,
        template_fallback: TemplateFallbackService,
        templates: dict[str, list[str]],
        template_fallback_actions: set[str],
        strategies: dict[str, Any],
    ) -> None:
        self._fact_repository = fact_repository
        self._template_fallback = template_fallback
        self._templates = templates
        self._template_fallback_actions = template_fallback_actions
        self._strategies = strategies

    def render(self, context: RenderContext) -> str | None:
        strategy = self._strategies.get(context.action_name)
        if strategy is not None:
            strategy_answer = strategy.render(context)
            if strategy_answer and not self._looks_like_meta_text(strategy_answer):
                return strategy_answer

        fact = self._fact_repository.find_best(
            topic_ids=context.topic_ids,
            action_name=context.action_name,
            user_query=context.user_query,
            history_text=context.history_text,
        )
        if fact is not None and fact.text and not self._looks_like_meta_text(fact.text):
            return fact.text
        return None

    @staticmethod
    def _looks_like_meta_text(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()
        if re.match(r"^\d+\.\s", normalized):
            return True
        return normalized.startswith("клиент ") or normalized.startswith("клиента ")
