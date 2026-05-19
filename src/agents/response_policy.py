from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.agents.policy.rule_engine import RuleEngine
from src.agents.policy.types import ResponseAction, ResponseState


def _load_response_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


@dataclass(slots=True)
class TopicActionRule:
    required_topics_any: set[str]
    required_topics_all: set[str]
    action: str
    secondary_topic: str = ""


@dataclass(slots=True)
class OrderedSelectorRule:
    all_flags: set[str]
    none_flags: set[str]
    topic_any: set[str]
    topic_all: set[str]
    current_focus_any: set[str]
    action: str
    secondary_topic: str = ""


@dataclass(slots=True)
class QueryUnderstanding:
    query: str
    history_lower: str
    primary_topic: str
    topic_set: set[str]
    closing_phrase: bool
    business_phrase: bool
    checkout_intent: bool
    api_fallback_purchase: bool
    legal_marker: bool
    legal_context_confirmed: bool
    price_query: bool
    parts_query: bool
    manager_query: bool
    manager_request_hits: int
    manager_phone_prompt: bool
    phone_digits: str
    phone_like_input: bool
    greeting_query: bool
    has_assistant_history: bool
    deescalation_query: bool
    compare_epc_tis_query: bool
    current_focus: str
    docs_query: bool
    out_of_scope_query: bool
    services_overview_query: bool


class ResponseActionSelector:
    def __init__(self, response_policy_file_path: Path) -> None:
        policy = _load_response_policy(response_policy_file_path)
        project_root = Path(__file__).resolve().parents[2]
        self._rule_engine = RuleEngine(project_root / "config/response_understanding_rules.yaml")
        selector = policy.get("selector", {}) if isinstance(policy.get("selector", {}), dict) else {}
        ordered_raw = selector.get("ordered_rules", [])
        self._ordered_rules: list[OrderedSelectorRule] = []
        if isinstance(ordered_raw, list):
            for item in ordered_raw:
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action", "")).strip()
                if not action:
                    continue
                self._ordered_rules.append(
                    OrderedSelectorRule(
                        all_flags={str(v).strip() for v in item.get("all_flags", []) if str(v).strip()},
                        none_flags={str(v).strip() for v in item.get("none_flags", []) if str(v).strip()},
                        topic_any={str(v).strip() for v in item.get("topic_any", []) if str(v).strip()},
                        topic_all={str(v).strip() for v in item.get("topic_all", []) if str(v).strip()},
                        current_focus_any={str(v).strip() for v in item.get("current_focus_any", []) if str(v).strip()},
                        action=action,
                        secondary_topic=str(item.get("secondary_topic", "")).strip(),
                    )
                )
        rules_raw = selector.get("topic_action_rules", [])
        self._topic_action_rules: list[TopicActionRule] = []
        if isinstance(rules_raw, list):
            for item in rules_raw:
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action", "")).strip()
                if not action:
                    continue
                self._topic_action_rules.append(
                    TopicActionRule(
                        required_topics_any={str(v).strip() for v in item.get("required_topics_any", []) if str(v).strip()},
                        required_topics_all={str(v).strip() for v in item.get("required_topics_all", []) if str(v).strip()},
                        action=action,
                        secondary_topic=str(item.get("secondary_topic", "")).strip(),
                    )
                )
        self._default_action = str(selector.get("default_action", "clarify_request")).strip() or "clarify_request"
        self._known_actions: set[str] = {self._default_action}
        self._known_actions.update(rule.action for rule in self._ordered_rules)
        self._known_actions.update(rule.action for rule in self._topic_action_rules)
        templates_raw = policy.get("templates", {})
        if isinstance(templates_raw, dict):
            self._known_actions.update(str(key).strip() for key in templates_raw.keys() if str(key).strip())

        allowed_raw = selector.get("allowed_actions_by_focus", {})
        self._allowed_actions_by_focus: dict[str, set[str]] = {}
        if isinstance(allowed_raw, dict):
            for focus, actions in allowed_raw.items():
                if not isinstance(actions, list):
                    continue
                parsed = {str(item).strip() for item in actions if str(item).strip()}
                if parsed:
                    self._allowed_actions_by_focus[str(focus).strip()] = parsed

    def select_from_planner(
        self,
        topic_ids: list[str],
        planned_action: str,
        current_focus: str,
    ) -> ResponseAction | None:
        action = str(planned_action).strip()
        if not action:
            return None
        if action not in self._known_actions:
            return None
        focus = str(current_focus).strip().lower()
        allowed = self._allowed_actions_by_focus.get(focus)
        if allowed is not None and action not in allowed:
            return None
        primary_topic = topic_ids[0] if topic_ids else "out_of_scope_request"
        return ResponseAction(action, primary_topic)

    def select(
        self,
        topic_ids: list[str],
        state: ResponseState,
        user_query: str,
        history_text: str = "",
        turn_analysis: dict[str, Any] | None = None,
    ) -> ResponseAction:
        topic_set = set(topic_ids)
        primary_topic = topic_ids[0] if topic_ids else "out_of_scope_request"
        query_data = self._understand(
            topic_set=topic_set,
            primary_topic=primary_topic,
            user_query=user_query,
            history_text=history_text,
            state=state,
            turn_analysis=turn_analysis,
        )
        action_by_ordered = self._select_by_ordered_rules(query_data=query_data, state=state, topic_set=topic_set, primary_topic=primary_topic)
        if action_by_ordered is not None:
            return action_by_ordered
        action_by_topic = self._select_by_topic_rules(topic_set, primary_topic)
        if action_by_topic is not None:
            return action_by_topic
        return ResponseAction(self._default_action, primary_topic)

    def _select_by_ordered_rules(
        self,
        query_data: QueryUnderstanding,
        state: ResponseState,
        topic_set: set[str],
        primary_topic: str,
    ) -> ResponseAction | None:
        flags = self._build_rule_flags(query_data, state, topic_set)
        for rule in self._ordered_rules:
            if rule.all_flags and not all(flags.get(flag, False) for flag in rule.all_flags):
                continue
            if rule.none_flags and any(flags.get(flag, False) for flag in rule.none_flags):
                continue
            if rule.topic_all and not rule.topic_all.issubset(topic_set):
                continue
            if rule.topic_any and not topic_set.intersection(rule.topic_any):
                continue
            if rule.current_focus_any and query_data.current_focus not in rule.current_focus_any:
                continue
            secondary = rule.secondary_topic if rule.secondary_topic else None
            return ResponseAction(rule.action, primary_topic, secondary)
        return None

    @staticmethod
    def _build_rule_flags(query_data: QueryUnderstanding, state: ResponseState, topic_set: set[str]) -> dict[str, bool]:
        return {
            "conversation_closed": state.conversation_closed,
            "closing_phrase": query_data.closing_phrase,
            "business_phrase": query_data.business_phrase,
            "not_greeted": not state.greeted,
            "no_assistant_history": not query_data.has_assistant_history,
            "greeting_query": query_data.greeting_query,
            "manager_phone_prompt": query_data.manager_phone_prompt,
            "phone_like_input": query_data.phone_like_input,
            "phone_digits_eq_11": len(query_data.phone_digits) == 11,
            "out_of_scope_query": query_data.out_of_scope_query,
            "deescalation_query": query_data.deescalation_query,
            "docs_query": query_data.docs_query,
            "services_overview_without_checkout": query_data.services_overview_query and not query_data.checkout_intent,
            "parts_query": query_data.parts_query,
            "manager_query_or_topic_human": query_data.manager_query or ("human_operator_request" in topic_set),
            "manager_request_hits_ge_2": query_data.manager_request_hits >= 2,
            "api_fallback_purchase": query_data.api_fallback_purchase,
            "checkout_intent": query_data.checkout_intent,
            "legal_context_confirmed": query_data.legal_context_confirmed,
        }

    def _select_by_topic_rules(self, topic_set: set[str], primary_topic: str) -> ResponseAction | None:
        for rule in self._topic_action_rules:
            if rule.required_topics_all and not rule.required_topics_all.issubset(topic_set):
                continue
            if rule.required_topics_any and not topic_set.intersection(rule.required_topics_any):
                continue
            secondary = rule.secondary_topic if rule.secondary_topic else None
            return ResponseAction(rule.action, primary_topic, secondary)
        return None

    def _understand(
        self,
        topic_set: set[str],
        primary_topic: str,
        user_query: str,
        history_text: str,
        state: ResponseState,
        turn_analysis: dict[str, Any] | None = None,
    ) -> QueryUnderstanding:
        query = user_query.lower().replace("ё", "е")
        history_lower = history_text.lower().replace("ё", "е")
        analysis = turn_analysis or {}
        analysis_signals = {str(item) for item in analysis.get("signals", []) if str(item).strip()}
        analysis_slots_raw = analysis.get("slots", {})
        analysis_slots = {str(key): bool(value) for key, value in analysis_slots_raw.items()} if isinstance(analysis_slots_raw, dict) else {}
        closing_phrase = self._rule_engine.match("closing_phrase", query)
        business_phrase = self._rule_engine.match("business_phrase", query)
        checkout_intent = self._rule_engine.match("checkout_intent", query)
        api_fallback_purchase = self._rule_engine.match("api_fallback_purchase", query)
        legal_marker = self._rule_engine.match("legal_marker", query)
        legal_context_confirmed = "legal_entity_purchase_flow" in topic_set or state.client_type == "legal" or legal_marker
        price_query = self._rule_engine.match("price_query", query)
        parts_query = bool(analysis.get("parts_query", False) or "parts_selection" in analysis_signals or self._rule_engine.match("parts_query", query))
        manager_query = bool(analysis.get("manager_query", False) or "human" in analysis_signals or self._rule_engine.match("manager_query", query))
        manager_request_hits = self._rule_engine.findall_count("manager_query", history_lower)
        last_assistant_line = self._extract_last_assistant(history_text).lower().replace("ё", "е")
        awaiting_phone = self._rule_engine.match("awaiting_phone", last_assistant_line)
        phone_validation_prompt = self._rule_engine.match("phone_validation_prompt", last_assistant_line)
        manager_in_history = self._rule_engine.match("manager_query", history_lower)
        manager_phone_prompt = (awaiting_phone and ("менедж" in last_assistant_line or manager_in_history or phone_validation_prompt)) or phone_validation_prompt
        phone_digits = self._extract_phone_digits(query)
        phone_like_input = self._is_phone_like_input(query, phone_digits)
        greeting_query = self._rule_engine.match("greeting_query", query.strip())
        has_assistant_history = self._rule_engine.match_multiline("has_assistant_history", history_lower)
        deescalation_query = self._rule_engine.match("deescalation_query", query)
        compare_epc_tis_query = self._rule_engine.match("compare_epc_tis_query", query)
        current_focus = str(analysis.get("current_focus", "")).strip().lower() or self._resolve_current_focus(query, topic_set, primary_topic, price_query, compare_epc_tis_query)
        docs_query = bool(
            analysis_slots.get("company_documents_card", False)
            or analysis.get("docs_query", False)
            or self._rule_engine.match("docs_query", query)
            or (self._rule_engine.match("docs_query", history_lower) and self._rule_engine.match("legal_marker", query))
        )
        out_of_scope_query = bool(
            analysis_slots.get("geo_map_request", False)
            or analysis_slots.get("out_of_scope_catalog", False)
            or analysis.get("out_of_scope_query", False)
            or "out_of_scope_catalog" in analysis_signals
            or self._rule_engine.match("out_of_scope_query", query)
        )
        if manager_phone_prompt and phone_like_input:
            out_of_scope_query = False
        services_overview_query = bool(
            analysis_slots.get("company_services_overview", False)
            or analysis.get("services_overview_query", False)
            or self._rule_engine.match("services_overview_query", query)
        )
        return QueryUnderstanding(
            query=query,
            history_lower=history_lower,
            primary_topic=primary_topic,
            topic_set=topic_set,
            closing_phrase=closing_phrase,
            business_phrase=business_phrase,
            checkout_intent=checkout_intent,
            api_fallback_purchase=api_fallback_purchase,
            legal_marker=legal_marker,
            legal_context_confirmed=legal_context_confirmed,
            price_query=price_query,
            parts_query=parts_query,
            manager_query=manager_query,
            manager_request_hits=manager_request_hits,
            manager_phone_prompt=manager_phone_prompt,
            phone_digits=phone_digits,
            phone_like_input=phone_like_input,
            greeting_query=greeting_query,
            has_assistant_history=has_assistant_history,
            deescalation_query=deescalation_query,
            compare_epc_tis_query=compare_epc_tis_query,
            current_focus=current_focus,
            docs_query=docs_query,
            out_of_scope_query=out_of_scope_query,
            services_overview_query=services_overview_query,
        )

    def _resolve_current_focus(
        self,
        query: str,
        topic_set: set[str],
        primary_topic: str,
        price_query: bool,
        compare_epc_tis_query: bool,
    ) -> str:
        return self._rule_engine.resolve_focus(
            query=query,
            topic_set=topic_set,
            primary_topic=primary_topic,
            price_query=price_query,
            compare_epc_tis_query=compare_epc_tis_query,
        )

    @staticmethod
    def _extract_last_assistant(history_text: str) -> str:
        lines = [line.strip() for line in history_text.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.lower().startswith("assistant:"):
                return line.split(":", 1)[1].strip()
        return ""

    @staticmethod
    def _extract_phone_digits(query: str) -> str:
        return "".join(ch for ch in query if ch.isdigit())

    def _is_phone_like_input(self, query: str, phone_digits: str) -> bool:
        if len(phone_digits) >= 5:
            return True
        return self._rule_engine.match("phone_like_keywords", query)


class ResponseContractValidator:
    def __init__(self, response_policy_file_path: Path) -> None:
        policy = _load_response_policy(response_policy_file_path)
        contracts_any_raw = policy.get("contracts_any")
        contracts_all_raw = policy.get("contracts_all")
        contracts_raw = policy.get("contracts", {})
        self._contracts_any: dict[str, list[str]] = self._parse_contracts(
            contracts_any_raw if isinstance(contracts_any_raw, dict) else contracts_raw
        )
        self._contracts_all: dict[str, list[str]] = self._parse_contracts(
            contracts_all_raw if isinstance(contracts_all_raw, dict) else {}
        )

    def validate(self, action: ResponseAction, answer_text: str) -> bool:
        normalized = answer_text.lower().replace("ё", "е")
        required_all = self._contracts_all.get(action.name, [])
        required_any = self._contracts_any.get(action.name, [])
        if required_all and not all(token in normalized for token in required_all):
            return False
        if required_any:
            return any(token in normalized for token in required_any)
        return True

    @staticmethod
    def _parse_contracts(raw_contracts: dict[str, object]) -> dict[str, list[str]]:
        parsed: dict[str, list[str]] = {}
        for action, tokens in raw_contracts.items():
            if not isinstance(tokens, list):
                continue
            normalized = [str(token).strip().lower() for token in tokens if str(token).strip()]
            if normalized:
                parsed[str(action).strip()] = normalized
        return parsed
