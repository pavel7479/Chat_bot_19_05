from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import re
import yaml

from src.app.product_resolver import ProductResolver
from src.app.service_response_progress_policy import ServiceResponseProgressPolicy
from src.core.models import TopicClassificationResult
from src.retrieval.fact_repository import FactRepository


@dataclass(slots=True)
class ResponsePlan:
    primary_action: str
    primary_topic: str
    locked_action: bool = False
    required_fact_ids: list[str] = field(default_factory=list)
    required_price_blocks: list[str] = field(default_factory=list)
    required_followup: list[str] = field(default_factory=list)
    response_mode: str = "knowledge"
    secondary_topic: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "primary_action": self.primary_action,
            "primary_topic": self.primary_topic,
            "locked_action": self.locked_action,
            "required_fact_ids": list(self.required_fact_ids),
            "required_price_blocks": list(self.required_price_blocks),
            "required_followup": list(self.required_followup),
            "response_mode": self.response_mode,
            "secondary_topic": self.secondary_topic,
        }


class ResponsePlanner:
    _GREETING_RE = re.compile(r"^\s*(привет|здравствуй|здравствуйте|добрый день|добрый вечер|hello|hi)\b", re.IGNORECASE)
    _TOPIC_TO_ACTION = {
        "epc_tariffs": "epc_tariffs",
        "tis_tariffs": "tis_tariffs",
        "specific_brand_check": "brand_availability",
        "brand_list_request": "brand_list_info",
        "catalog_list_request": "catalog_list_info",
        "company_services_info": "company_services",
        "self_employed_purchase": "self_employed_policy",
        "existing_contract_check": "existing_contract_check",
        "subscription_renewal": "subscription_renewal",
        "payment_process": "payment_process",
        "competitor_comparison": "competitor_comparison",
        "demo_access": "demo_policy",
        "purchase_ready": "ask_legal_status",
        "legal_entity_purchase_flow": "request_requisites",
        "payment_without_details": "request_requisites",
        "api_integration": "api_unavailable",
        "human_operator_request": "human_operator",
        "physical_person_purchase": "physical_reject",
        "no_private_sales_reason": "physical_reject_reason",
        "price_objection": "price_objection",
        "competitor_choice": "price_objection",
        "partial_catalog_request": "partial_catalog_restriction",
        "free_catalog_comparison": "free_catalog_comparison",
        "product_relation_or_difference": "compare_epc_tis",
        "macos_support": "macos_support",
        "post_payment_access_timing": "post_payment_access_info",
        "post_payment_no_access": "post_payment_no_access_handoff",
        "usage_limits": "usage_limits_info",
        "multi_device_access": "multi_device_access_info",
        "manager_setup_support": "manager_setup_support_info",
        "nonsense_input": "clarify_request",
        "out_of_scope_request": "out_of_scope_response",
    }
    _LOCKED_ACTIONS = {
        "human_operator",
        "ask_legal_status",
        "brand_group_clarification",
        "partial_catalog_restriction",
        "greeting_once",
    }

    def __init__(self, fact_map_file_path: Path, brands_file_path: Path, facts_file_path: Path | None = None) -> None:
        self._fact_map = self._load_fact_map(fact_map_file_path)
        self._product_resolver = ProductResolver(brands_file_path)
        self._fact_repository = FactRepository(facts_file_path) if facts_file_path is not None else None
        self._service_progress_policy = ServiceResponseProgressPolicy()

    def plan(
        self,
        topic_result: TopicClassificationResult,
        user_query: str,
        history_text: str = "",
    ) -> ResponsePlan:
        topic_ids = [str(topic).strip() for topic in topic_result.topic_ids if str(topic).strip()]
        topic_set = set(topic_ids)
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        slot_trace = diagnostics.get("slot_extraction_trace", {}) if isinstance(diagnostics.get("slot_extraction_trace", {}), dict) else {}
        product_context = self._product_resolver.resolve(
            user_query=user_query,
            history_text=history_text,
            state_snapshot=topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {},
            slot_trace=slot_trace,
        )
        merged_slots = self._merged_slots(topic_result)
        state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        service_memory = state_snapshot.get("service_semantic_memory", {}) if isinstance(state_snapshot.get("service_semantic_memory", {}), dict) else {}
        if isinstance(service_memory, dict):
            if "used_fact_ids" in service_memory and "used_service_fact_ids" not in merged_slots:
                merged_slots["used_service_fact_ids"] = list(service_memory.get("used_fact_ids", []))
            if "used_groups" in service_memory and "used_service_semantic_groups" not in merged_slots:
                merged_slots["used_service_semantic_groups"] = list(service_memory.get("used_groups", []))
        diagnostics["product_context_trace"] = product_context.as_dict()
        diagnostics["conversation_reasoning_trace"] = {
            **dict(diagnostics.get("conversation_reasoning_trace", {})),
            "product_context": product_context.as_dict(),
            "business_flow": str(state_snapshot.get("active_business_flow", "none")),
        }
        topic_result.diagnostics = diagnostics

        action, primary_topic, secondary_topic = self._resolve_action(
            topic_ids=topic_ids,
            topic_set=topic_set,
            topic_result=topic_result,
            product_context=product_context,
            merged_slots=merged_slots,
        )
        fact_ids = self._resolve_fact_ids(
            action=action,
            topic_ids=topic_ids,
            topic_set=topic_set,
            user_query=user_query,
            history_text=history_text,
            product_context=product_context,
            merged_slots=merged_slots,
        )
        price_blocks = self._resolve_price_blocks(
            action=action,
            topic_set=topic_set,
            product_context=product_context,
        )
        diagnostics["planner_trace"] = {
            "selected_action": {
                "name": action,
                "locked_action": action in self._LOCKED_ACTIONS,
            },
            "current_flow": str(state_snapshot.get("active_business_flow", "none")),
            "pricing_state": dict(state_snapshot.get("pricing_flow", {})) if isinstance(state_snapshot.get("pricing_flow", {}), dict) else {},
            "service_semantic_memory": dict(state_snapshot.get("service_semantic_memory", {})) if isinstance(state_snapshot.get("service_semantic_memory", {}), dict) else {},
            "pending_brands": list(state_snapshot.get("pending_brand_mentions", [])) if isinstance(state_snapshot.get("pending_brand_mentions", []), list) else [],
        }
        return ResponsePlan(
            primary_action=action,
            primary_topic=primary_topic,
            locked_action=action in self._LOCKED_ACTIONS,
            required_fact_ids=fact_ids,
            required_price_blocks=price_blocks,
            required_followup=[],
            response_mode="knowledge",
            secondary_topic=secondary_topic,
        )

    @staticmethod
    def _load_fact_map(path: Path) -> dict[str, dict[str, list[str]]]:
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            return {}
        parsed: dict[str, dict[str, list[str]]] = {}
        for action, variants in raw.items():
            if not isinstance(variants, dict):
                continue
            action_name = str(action).strip()
            if not action_name:
                continue
            parsed[action_name] = {}
            for variant, fact_ids in variants.items():
                if not isinstance(fact_ids, list):
                    continue
                variant_name = str(variant).strip()
                if not variant_name:
                    continue
                parsed[action_name][variant_name] = [str(item).strip() for item in fact_ids if str(item).strip()]
        return parsed

    def _resolve_action(
        self,
        *,
        topic_ids: list[str],
        topic_set: set[str],
        topic_result: TopicClassificationResult,
        product_context,
        merged_slots: dict[str, object],
    ) -> tuple[str, str, str | None]:
        state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        dialog_act_trace = diagnostics.get("dialog_act_trace", {})
        if isinstance(dialog_act_trace, dict):
            hard_action = str(dialog_act_trace.get("action_name", "")).strip()
            hard_topics = dialog_act_trace.get("topic_ids", [])
            if hard_action:
                primary_topic = str(hard_topics[0]).strip() if isinstance(hard_topics, list) and hard_topics else (topic_ids[0] if topic_ids else topic_result.primary_topic_id)
                return hard_action, primary_topic, None
        already_greeted = bool(state_snapshot.get("greeted", False))
        raw_query = str(getattr(product_context, "raw_query", "") or "")
        non_service_topics = topic_set - {"nonsense_input", "out_of_scope_request"}
        if (
            not already_greeted
            and self._GREETING_RE.search(raw_query)
            and not non_service_topics
            and not product_context.asks_price
            and not product_context.asks_availability
            and not product_context.asks_comparison
        ):
            return "greeting_once", "nonsense_input", None

        if product_context.asks_comparison or "product_relation_or_difference" in topic_set:
            return "compare_epc_tis", "product_relation_or_difference", None

        for explicit_topic in (
            "existing_contract_check",
            "self_employed_purchase",
            "subscription_renewal",
            "payment_process",
            "competitor_comparison",
        ):
            if explicit_topic in topic_set:
                return self._TOPIC_TO_ACTION[explicit_topic], explicit_topic, None

        if product_context.unsupported_brand_group and ("specific_brand_check" in topic_set or "partial_catalog_request" in topic_set):
            return "brand_group_clarification", "specific_brand_check", None

        if product_context.unknown_brand_query and ("specific_brand_check" in topic_set or product_context.asks_availability):
            return "unknown_brand_unavailable", "specific_brand_check", None

        if (
            ("partial_catalog_request" in topic_set or product_context.partial_access_requested or product_context.partial_package_requested or getattr(product_context, "partial_epc_requested", False))
            and not product_context.mentioned_brands
        ):
            return "partial_catalog_restriction", "partial_catalog_request", None

        if self._checkout_ready_for_manager(topic_set=topic_set, merged_slots=merged_slots, state_snapshot=state_snapshot):
            return "manager_handoff_ready", "legal_entity_purchase_flow", None

        if product_context.asks_price and product_context.inferred_product_context == "tis" and product_context.mentioned_brands:
            return "tis_tariffs", "tis_tariffs", None

        if product_context.inferred_product_context == "tis" and product_context.mentioned_brands:
            return "brand_availability", "specific_brand_check", None

        if product_context.asks_price and not product_context.requested_products and ({"epc_tariffs", "tis_tariffs"} & topic_set or "purchase_ready" in topic_set):
            return "pricing_summary", "epc_tariffs", "tis_tariffs"

        if {"epc_tariffs", "tis_tariffs"}.issubset(topic_set):
            return "pricing_summary", "epc_tariffs", "tis_tariffs"

        non_fallback_topics = [topic for topic in topic_ids if topic not in {"nonsense_input", "out_of_scope_request"}]
        if non_fallback_topics:
            for topic_key in non_fallback_topics:
                if topic_key in self._TOPIC_TO_ACTION:
                    return self._TOPIC_TO_ACTION[topic_key], topic_key, None
        for topic_id in topic_ids:
            if topic_id in self._TOPIC_TO_ACTION:
                return self._TOPIC_TO_ACTION[topic_id], topic_result.primary_topic_id, None
        return "clarify_request", topic_result.primary_topic_id, None

    def _resolve_fact_ids(
        self,
        *,
        action: str,
        topic_ids: list[str],
        topic_set: set[str],
        user_query: str,
        history_text: str,
        product_context,
        merged_slots: dict[str, object],
    ) -> list[str]:
        variants = self._fact_map.get(action, {})
        ordered: list[str] = []

        def extend(variant_name: str) -> None:
            for fact_id in variants.get(variant_name, []):
                if fact_id not in ordered:
                    ordered.append(fact_id)

        extend("default")
        query = self._normalize(user_query)
        history = self._normalize(history_text)
        combined = f"{history} {query}".strip()

        if action == "company_services":
            used_service_fact_ids = merged_slots.get("used_service_fact_ids", [])
            if not isinstance(used_service_fact_ids, list):
                used_service_fact_ids = []
            used_service_semantic_groups = merged_slots.get("used_service_semantic_groups", [])
            if not isinstance(used_service_semantic_groups, list):
                used_service_semantic_groups = []
            service_progression = [
                "company_services_general",
                "company_services_competitor_discount",
                "company_services_cons",
                "catalog_list_products",
            ]
            ordered.clear()
            next_fact_ids = (
                self._service_progress_policy.select_next_fact_ids(
                    candidate_fact_ids=service_progression,
                    used_fact_ids=used_service_fact_ids,
                    used_semantic_groups=used_service_semantic_groups,
                    fact_catalog=self._fact_repository,
                )
                if self._fact_repository is not None
                else [service_progression[-1]]
            )
            ordered.extend(next_fact_ids)
            if any(token in combined for token in ("огранич", "минус", "неудоб", "рамк")):
                extend("limitations")
            if any(token in combined for token in ("компьют", "устройств", "mac", "ноут")):
                extend("multi_device")
            if any(token in combined for token in ("нескольк", "сотрудник", "пользоват", "доступов")):
                extend("multi_user")
            if any(token in combined for token in ("подбер", "артикул", "запчаст", "детал")):
                extend("parts_selection")
            if any(token in combined for token in ("лимит", "ограничени", "запросов", "безлимит")):
                extend("usage_limits")
        elif action == "brand_availability":
            ordered.clear()
            if product_context.mentioned_brands:
                extend("specific")
            else:
                extend("need_brand")
        elif action == "brand_group_clarification":
            ordered.clear()
            extend("default")
        elif action == "unknown_brand_unavailable":
            ordered.clear()
            extend("default")
        elif action == "manager_handoff_ready":
            ordered.clear()
            extend("default")
        elif action in {"brand_list_info", "catalog_list_info", "existing_contract_check", "self_employed_policy", "subscription_renewal", "payment_process", "competitor_comparison"}:
            ordered.clear()
            extend("default")
        elif action == "tis_tariffs":
            ordered.clear()
            if product_context.mentioned_brands:
                extend("known_brand")
            else:
                extend("default")
        elif action == "pricing_summary":
            ordered.clear()
            extend("default")
        elif action == "demo_policy":
            if "legal_entity_purchase_flow" in topic_set or any(token in query for token in ("являюсь", "yes", "legal", "юр", "ип")):
                extend("confirmed_legal")
        elif action == "human_operator":
            if any(token in combined for token in ("мошен", "туп", "развод", "обман", "жалоб")):
                extend("abuse")
        elif action == "compare_epc_tis":
            if "free_catalog_comparison" in topic_set and "product_relation_or_difference" not in topic_set:
                ordered.clear()
                extend("free_catalog")
        elif action == "request_requisites" and merged_slots:
            ordered.clear()
            extend("default")
        return ordered

    @staticmethod
    def _resolve_price_blocks(*, action: str, topic_set: set[str], product_context) -> list[str]:
        blocks: list[str] = []
        if action in {"epc_tariffs", "pricing_summary"} or ("epc_tariffs" in topic_set and action not in {"tis_tariffs", "ask_legal_status"}):
            blocks.append("epc")
        if action in {"tis_tariffs", "pricing_summary"} or ("tis_tariffs" in topic_set and action != "ask_legal_status"):
            blocks.append("tis")
        return blocks

    @staticmethod
    def _normalize(text: str) -> str:
        return str(text or "").lower().replace("ё", "е")

    @staticmethod
    def _merged_slots(topic_result: TopicClassificationResult) -> dict[str, object]:
        state_snapshot = topic_result.state_snapshot if isinstance(topic_result.state_snapshot, dict) else {}
        base_slots = state_snapshot.get("slots", {}) if isinstance(state_snapshot.get("slots", {}), dict) else {}
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        slot_trace = diagnostics.get("slot_extraction_trace", {}) if isinstance(diagnostics.get("slot_extraction_trace", {}), dict) else {}
        extracted_slots = slot_trace.get("slots", {}) if isinstance(slot_trace.get("slots", {}), dict) else {}
        merged = dict(base_slots)
        merged.update(extracted_slots)
        return merged

    @staticmethod
    def _checkout_ready_for_manager(
        *,
        topic_set: set[str],
        merged_slots: dict[str, object],
        state_snapshot: dict[str, object],
    ) -> bool:
        if "legal_entity_purchase_flow" not in topic_set:
            return False
        phone = str(merged_slots.get("phone", "")).strip()
        inn = str(merged_slots.get("inn", "")).strip()
        period = str(merged_slots.get("period", "")).strip()
        payment_method = str(merged_slots.get("payment_method", "")).strip()
        legal_status = str(merged_slots.get("legal_status", "")).strip() or str(state_snapshot.get("slots", {}).get("legal_status", "")).strip()
        if phone and inn and period and payment_method:
            return True
        if phone and legal_status in {"ip", "legal_entity"} and str(state_snapshot.get("active_request_kind", "")).strip() == "checkout":
            return True
        return False
