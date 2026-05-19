from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.agents.policy.response_render_service import (
    BrandAvailabilityStrategy,
    CheckoutStrategy,
    ComparisonStrategy,
    OutOfScopeStrategy,
    PricingStrategy,
    RenderContext,
    ResponseRenderService,
    TemplateFallbackService,
)
from src.agents.policy.types import ResponseAction, ResponseState
from src.domain.brands import BrandAliasResolver
from src.domain.pricing import TisPriceCatalog
from src.retrieval.fact_repository import FactRepository


def _load_response_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


@dataclass(slots=True)
class FactFirstStrategy:
    action_name: str

    def render(self, context: RenderContext) -> str | None:
        if context.action_name != self.action_name:
            return None
        return None


class DeterministicResponseBuilder:
    def __init__(
        self,
        brand_resolver: BrandAliasResolver,
        tis_prices: TisPriceCatalog,
        fact_repository: FactRepository,
        response_policy_file_path: Path,
    ) -> None:
        self._brand_resolver = brand_resolver
        self._tis_prices = tis_prices
        self._fact_repository = fact_repository
        policy = _load_response_policy(response_policy_file_path)
        self._templates: dict[str, list[str]] = {}
        self._template_fallback_actions: set[str] = set()
        self._template_fallback = TemplateFallbackService()
        strategy_registry = self._build_strategy_registry(policy)
        self._render_service = ResponseRenderService(
            fact_repository=self._fact_repository,
            template_fallback=self._template_fallback,
            templates=self._templates,
            template_fallback_actions=self._template_fallback_actions,
            strategies=strategy_registry,
        )

    def _build_strategy_registry(self, policy: dict[str, Any]) -> dict[str, Any]:
        specific: dict[str, Any] = {
            "tis_tariffs": PricingStrategy(brand_resolver=self._brand_resolver, tis_prices=self._tis_prices),
            "brand_availability": BrandAvailabilityStrategy(brand_resolver=self._brand_resolver, tis_prices=self._tis_prices),
            "compare_epc_tis": ComparisonStrategy(),
            "request_requisites": CheckoutStrategy(),
            "ask_legal_status": CheckoutStrategy(),
            "out_of_scope_response": OutOfScopeStrategy(),
        }
        known_actions = self._collect_known_actions(policy)
        registry: dict[str, Any] = {}
        for action in known_actions:
            registry[action] = specific.get(action, FactFirstStrategy(action_name=action))
        return registry

    @staticmethod
    def _collect_known_actions(policy: dict[str, Any]) -> set[str]:
        known: set[str] = set()
        selector = policy.get("selector", {}) if isinstance(policy.get("selector", {}), dict) else {}
        default_action = str(selector.get("default_action", "")).strip()
        if default_action:
            known.add(default_action)
        for rule in selector.get("ordered_rules", []) if isinstance(selector.get("ordered_rules", []), list) else []:
            if isinstance(rule, dict):
                action = str(rule.get("action", "")).strip()
                if action:
                    known.add(action)
        for rule in selector.get("topic_action_rules", []) if isinstance(selector.get("topic_action_rules", []), list) else []:
            if isinstance(rule, dict):
                action = str(rule.get("action", "")).strip()
                if action:
                    known.add(action)
        templates = policy.get("templates", {}) if isinstance(policy.get("templates", {}), dict) else {}
        known.update(str(action).strip() for action in templates.keys() if str(action).strip())
        contracts = policy.get("contracts", {}) if isinstance(policy.get("contracts", {}), dict) else {}
        known.update(str(action).strip() for action in contracts.keys() if str(action).strip())
        return known

    def build(
        self,
        action: ResponseAction,
        history_text: str,
        user_query: str,
        topic_ids: list[str],
        state: ResponseState,
    ) -> str | None:
        rendered = self._render_service.render(
            context=RenderContext(
                action_name=action.name,
                history_text=history_text,
                user_query=user_query,
                topic_ids=topic_ids,
                state=state,
            )
        )
        recent_answers = self._template_fallback._recent_assistant_answers(history_text, limit=2)
        if rendered is not None:
            normalized_rendered = self._template_fallback._normalize_answer(rendered)
            normalized_recent = {self._template_fallback._normalize_answer(item) for item in recent_answers}
            if normalized_rendered not in normalized_recent:
                return rendered
        return rendered

    def build_template(
        self,
        action: ResponseAction,
        history_text: str,
        user_query: str,
        topic_ids: list[str],
        state: ResponseState,
    ) -> str | None:
        return self._render_service.render(
            context=RenderContext(
                action_name=action.name,
                history_text=history_text,
                user_query=user_query,
                topic_ids=topic_ids,
                state=state,
            )
        )
