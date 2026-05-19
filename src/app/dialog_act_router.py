from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.app.product_resolver import ProductContext
from src.core.models import SessionState


@dataclass(slots=True)
class DialogActDecision:
    applied: bool
    reason: str
    topic_ids: list[str]
    action_name: str
    classifier_source: str
    active_request_kind: str = ""
    extra_state_patch: dict[str, object] = field(default_factory=dict)
    trace: dict[str, object] = field(default_factory=dict)


class DialogActRouter:
    _DEMO_WORDS = ("демо", "пробный доступ", "тестовый доступ")
    _SHORT_POSITIVE_REPLIES = {"да", "ага", "угу", "ок", "okay", "yes", "yep"}
    _SERVICE_CONTINUE_REPLIES = {
        "ну попробуй",
        "попробуй",
        "рассказывай",
        "да",
        "давай",
        "ну давай",
    }
    _SERVICE_PREVIOUS_ACTIONS = {"greeting_once", "company_services"}
    _SEPARATE_PRICING_RE = re.compile(r"^\s*(отдельно|по отдельности|каждый отдельно|по каждому отдельно)\s*$", re.IGNORECASE)
    _REMAINING_BRANDS_RE = re.compile(r"^\s*(а остальные|остальные|а по другим|а еще)\s*$", re.IGNORECASE)
    _PUSHBACK_RE = re.compile(
        r"^\s*(что уточнить|это ты мне уточни|что именно уточнить|ты должен уточнить|ты сам уточни|сколько можно)\s*$",
        re.IGNORECASE,
    )
    _IRRITATED_PUSHBACK_RE = re.compile(
        r"^\s*(это ты мне уточни|ты сам уточни|сколько можно)\s*$",
        re.IGNORECASE,
    )
    _PRICE_TOKENS = ("сколько", "стоимость", "цена", "тариф", "по деньгам")
    _PURCHASE_FLOW_PATTERNS = (
        re.compile(r"\bкак купить подписку\b", re.IGNORECASE),
        re.compile(r"\bкак оформить подписку\b", re.IGNORECASE),
        re.compile(r"\bкак подключить доступ\b", re.IGNORECASE),
        re.compile(r"\bкак приобрести\b", re.IGNORECASE),
    )

    def route(
        self,
        *,
        user_query: str,
        session_state: SessionState,
        context_result,
        slot_trace: dict[str, object],
        product_context: ProductContext,
    ) -> DialogActDecision:
        normalized_query = self._normalize(user_query)
        turn_type = str(getattr(context_result, "turn_type", "") or "").strip()
        slots = slot_trace.get("slots", {}) if isinstance(slot_trace.get("slots", {}), dict) else {}

        if turn_type == "greeting" and not session_state.greeted:
            return self._decision(
                reason="greeting_short_circuit",
                topic_ids=["nonsense_input"],
                action_name="greeting_once",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
                active_request_kind="company_services",
                extra_state_patch={
                    "greeted": True,
                    "last_action_name": "greeting_once",
                    "dialog_phase": "discovery",
                    "active_business_flow": "company_services",
                },
            )

        if product_context.unsupported_brand_group == "VAG":
            return self._decision(
                reason="unsupported_brand_group",
                topic_ids=["specific_brand_check"],
                action_name="brand_group_clarification",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
            )

        if self._is_purchase_flow(normalized_query):
            return self._decision(
                reason="purchase_flow_without_price_request",
                topic_ids=["purchase_ready"],
                action_name="ask_legal_status",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
                active_request_kind="checkout",
                extra_state_patch={
                    "active_business_flow": "purchase_flow",
                },
            )

        if self._is_service_continuation(
            normalized_query=normalized_query,
            turn_type=turn_type,
            previous_action=session_state.last_action_name,
            active_business_flow=session_state.active_business_flow,
        ):
            return self._decision(
                reason="service_continuation",
                topic_ids=["company_services_info"],
                action_name="company_services",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
                active_request_kind="company_services",
                extra_state_patch={
                    "active_business_flow": "company_services",
                },
            )

        raw_mentions = slots.get("raw_brand_mentions", [])
        has_raw_mentions = isinstance(raw_mentions, list) and bool(raw_mentions)
        if turn_type == "brand_list_for_tis" or (
            session_state.active_pricing_flow == "tis" and has_raw_mentions
        ):
            return self._decision(
                reason="tis_brand_list_pricing_flow",
                topic_ids=["tis_tariffs"],
                action_name="tis_tariffs",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
                active_request_kind="tis_tariffs",
                extra_state_patch={
                    "active_pricing_flow": "tis",
                    "pricing_requested_product": "tis",
                    "pricing_mode": "all",
                    "active_business_flow": "pricing_tis",
                },
            )

        if self._SEPARATE_PRICING_RE.match(user_query) and (
            session_state.active_pricing_flow == "tis" or session_state.active_business_flow == "pricing_tis"
        ):
            return self._decision(
                reason="separate_pricing_requested",
                topic_ids=["tis_tariffs"],
                action_name="tis_tariffs",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
                active_request_kind="tis_tariffs",
                extra_state_patch={
                    "active_pricing_flow": "tis",
                    "pricing_requested_product": "tis",
                    "pricing_mode": "separate_processing",
                    "active_business_flow": "pricing_tis",
                },
            )

        has_unresolved = bool(session_state.unknown_brand_mentions or session_state.missing_price_brands or session_state.pending_brand_mentions)
        if self._REMAINING_BRANDS_RE.match(user_query) and has_unresolved:
            return self._decision(
                reason="remaining_brands_followup",
                topic_ids=["tis_tariffs"],
                action_name="tis_tariffs",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
                active_request_kind="tis_tariffs",
                extra_state_patch={
                    "active_pricing_flow": "tis",
                    "pricing_requested_product": "tis",
                    "pricing_mode": "remaining_only",
                    "active_business_flow": "pricing_tis",
                },
            )

        if self._PUSHBACK_RE.match(user_query) and (
            session_state.last_bot_question_type == "pricing_clarification"
            or session_state.active_pricing_flow == "tis"
            or session_state.active_business_flow == "pricing_tis"
        ):
            if self._IRRITATED_PUSHBACK_RE.match(user_query):
                return self._decision(
                    reason="clarification_pushback_handoff",
                    topic_ids=["human_operator_request"],
                    action_name="human_operator",
                    session_state=session_state,
                    context_result=context_result,
                    slot_trace=slot_trace,
                    product_context=product_context,
                    active_request_kind="human_operator",
                    extra_state_patch={
                        "active_business_flow": "manager_handoff",
                    },
                )
            if has_unresolved:
                return self._decision(
                    reason="clarification_pushback_explain_unresolved",
                    topic_ids=["tis_tariffs"],
                    action_name="tis_tariffs",
                    session_state=session_state,
                    context_result=context_result,
                    slot_trace=slot_trace,
                    product_context=product_context,
                    active_request_kind="tis_tariffs",
                    extra_state_patch={
                        "active_pricing_flow": "tis",
                        "pricing_requested_product": "tis",
                        "pricing_mode": "explain_unresolved",
                        "active_business_flow": "pricing_tis",
                    },
                )
            return self._decision(
                reason="clarification_pushback_handoff_no_unresolved",
                topic_ids=["human_operator_request"],
                action_name="human_operator",
                session_state=session_state,
                context_result=context_result,
                slot_trace=slot_trace,
                product_context=product_context,
                active_request_kind="human_operator",
                extra_state_patch={
                    "active_business_flow": "manager_handoff",
                },
            )

        return DialogActDecision(
            applied=False,
            reason="no_hard_dialog_act",
            topic_ids=[],
            action_name="",
            classifier_source="",
        )

    def _decision(
        self,
        *,
        reason: str,
        topic_ids: list[str],
        action_name: str,
        session_state: SessionState,
        context_result,
        slot_trace: dict[str, object],
        product_context: ProductContext,
        active_request_kind: str = "",
        extra_state_patch: dict[str, object] | None = None,
    ) -> DialogActDecision:
        patch = dict(extra_state_patch or {})
        patch.setdefault("last_primary_topic", topic_ids[0] if topic_ids else "out_of_scope_request")
        patch.setdefault("last_topic_ids", list(topic_ids))
        patch.setdefault("last_secondary_topics", [])
        patch.setdefault("last_focus_topic", topic_ids[0] if topic_ids else "out_of_scope_request")
        if active_request_kind:
            patch.setdefault("active_request_kind", active_request_kind)
        trace = {
            "applied": True,
            "reason": reason,
            "topic_ids": list(topic_ids),
            "action_name": action_name,
            "turn_type": str(getattr(context_result, "turn_type", "") or ""),
            "turn_subtype": str(getattr(context_result, "turn_subtype", "") or ""),
            "user_query": str(getattr(product_context, "raw_query", "") or ""),
            "slot_snapshot": dict(slot_trace.get("slots", {})) if isinstance(slot_trace.get("slots", {}), dict) else {},
            "product_context": product_context.as_dict(),
            "state_before": session_state.as_dict(),
            "extra_state_patch": dict(patch),
        }
        return DialogActDecision(
            applied=True,
            reason=reason,
            topic_ids=list(topic_ids),
            action_name=action_name,
            classifier_source="dialog_act_router",
            active_request_kind=active_request_kind,
            extra_state_patch=patch,
            trace=trace,
        )

    def _is_purchase_flow(self, normalized_query: str) -> bool:
        if any(pattern.search(normalized_query) for pattern in self._PURCHASE_FLOW_PATTERNS):
            return not self._contains_any(normalized_query, self._PRICE_TOKENS)
        return False

    def _is_service_continuation(
        self,
        *,
        normalized_query: str,
        turn_type: str,
        previous_action: str,
        active_business_flow: str = "",
    ) -> bool:
        if self._contains_any(normalized_query, self._DEMO_WORDS):
            return False
        previous_action = str(previous_action or "").strip()
        if previous_action not in self._SERVICE_PREVIOUS_ACTIONS and active_business_flow != "company_services":
            return False
        if normalized_query in self._SERVICE_CONTINUE_REPLIES:
            return True
        return turn_type == "service_discovery_continue" and (
            previous_action == "company_services" or active_business_flow == "company_services"
        )

    @staticmethod
    def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
        return any(token in text for token in tokens)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()
