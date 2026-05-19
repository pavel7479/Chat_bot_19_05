from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.models import ContextSignals, SessionState


@dataclass(slots=True)
class FollowupResolution:
    is_followup: bool = False
    followup_type: str = "unknown_followup"
    inherited_brand: str = ""
    inherited_topics: list[str] = field(default_factory=list)
    suggested_topics: list[str] = field(default_factory=list)
    reason: str = ""
    trace: dict[str, object] = field(default_factory=dict)


class FollowupResolver:
    _SHORT_PRICE_RE = re.compile(r"^\s*(подскажи|подсказывай|давай|интересно|сколько|цена)\s*$", re.IGNORECASE)
    _SERVICE_CONTINUE_RE = re.compile(r"^\s*(ну попробуй|рассказывай|да)\s*$", re.IGNORECASE)
    _SEPARATE_PRICING_RE = re.compile(r"^\s*(отдельно|по отдельности|каждый отдельно|по каждому отдельно)\s*$", re.IGNORECASE)
    _REMAINING_BRANDS_RE = re.compile(r"^\s*(а остальные|остальные|а по другим|а еще)\s*$", re.IGNORECASE)
    _ONLY_BRAND_RE = re.compile(r"^\s*только\b", re.IGNORECASE)
    _PUSHBACK_RE = re.compile(
        r"^\s*(что уточнить|это ты мне уточни|что именно уточнить|ты должен уточнить|ты сам уточни|сколько можно)\s*$",
        re.IGNORECASE,
    )
    _PHONE_ONLY_RE = re.compile(r"^\s*(?:\+7|8)?[\d\s() -]{10,18}\s*$")
    _CONTRACT_RE = re.compile(r"\b(?:договор|номер договора|существующий договор|проверить договор)\b", re.IGNORECASE)

    def resolve(
        self,
        user_query: str,
        session_state: SessionState,
        context_signals: ContextSignals | None,
        slot_trace: dict[str, object],
        agent_zero_turn_type: str = "",
    ) -> FollowupResolution:
        query = str(user_query or "").strip()
        normalized = self._normalize(query)
        slots = slot_trace.get("slots", {}) if isinstance(slot_trace, dict) else {}
        inherited_topics = list(session_state.last_topic_ids)
        inherited_brand = str(session_state.last_mentioned_brand).strip().lower()
        turn_type = str(agent_zero_turn_type or "").strip()

        if (
            turn_type == "service_discovery_continue"
            and session_state.last_action_name == "company_services"
            and self._SERVICE_CONTINUE_RE.match(query)
        ):
            return self._build(
                is_followup=True,
                followup_type="service_continue_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["company_services_info"],
                reason="Service continuation detected after company services answer.",
                session_state=session_state,
                user_query=query,
            )

        if (
            turn_type == "brand_list_for_tis"
            and session_state.last_action_name in {"pricing_summary", "tis_tariffs"}
        ):
            return self._build(
                is_followup=True,
                followup_type="tis_brand_list_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["tis_tariffs", "specific_brand_check"],
                reason="Brand list for TIS detected after pricing request.",
                session_state=session_state,
                user_query=query,
            )

        if (
            turn_type == "pricing_followup"
            and self._SEPARATE_PRICING_RE.match(query)
            and session_state.active_pricing_flow == "tis"
        ):
            return self._build(
                is_followup=True,
                followup_type="separate_pricing_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["tis_tariffs"],
                reason="User requested separate per-brand pricing in an active TIS flow.",
                session_state=session_state,
                user_query=query,
            )

        if (
            turn_type == "pricing_followup"
            and self._REMAINING_BRANDS_RE.match(query)
            and session_state.active_pricing_flow == "tis"
        ):
            return self._build(
                is_followup=True,
                followup_type="remaining_brands_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["tis_tariffs"],
                reason="User asked about remaining brands in an active TIS pricing flow.",
                session_state=session_state,
                user_query=query,
            )

        if (
            turn_type in {"clarification_pushback", "human_escalation"}
            and self._PUSHBACK_RE.match(query)
            and session_state.last_bot_question_type == "pricing_clarification"
        ):
            return self._build(
                is_followup=True,
                followup_type="clarification_pushback_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["tis_tariffs", "human_operator_request"],
                reason="User pushed back on pricing clarification.",
                session_state=session_state,
                user_query=query,
            )

        if self._CONTRACT_RE.search(query):
            return self._build(
                is_followup=True,
                followup_type="contract_check_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["existing_contract_check"],
                reason="Contract-related wording detected in user query.",
                session_state=session_state,
                user_query=query,
            )

        if (
            inherited_brand
            and "specific_brand_check" in inherited_topics
            and session_state.last_action_name == "brand_availability"
            and self._SHORT_PRICE_RE.match(query)
        ):
            return self._build(
                is_followup=True,
                followup_type="brand_price_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["tis_tariffs", "specific_brand_check"],
                reason="Short follow-up after brand availability answer with known brand.",
                session_state=session_state,
                user_query=query,
            )

        if (
            inherited_brand
            and "specific_brand_check" in inherited_topics
            and session_state.last_action_name == "brand_availability"
            and self._ONLY_BRAND_RE.match(query)
            and self._slot_matches_inherited_brand(slots=slots, inherited_brand=inherited_brand)
        ):
            return self._build(
                is_followup=True,
                followup_type="brand_price_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["tis_tariffs", "specific_brand_check"],
                reason="Brand-only confirmation after availability answer keeps inherited brand pricing follow-up.",
                session_state=session_state,
                user_query=query,
            )

        if (
            session_state.last_bot_question_type in {"purchase_confirmation", "request_requisites", "legal_status_check"}
            and isinstance(slots, dict)
            and any(name in slots for name in ("phone", "inn", "payment_method", "period"))
        ):
            return self._build(
                is_followup=True,
                followup_type="requisites_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["legal_entity_purchase_flow"],
                reason="Reply contains requisites after checkout/requisites prompt.",
                session_state=session_state,
                user_query=query,
            )

        if (
            self._PHONE_ONLY_RE.match(query)
            and isinstance(slots, dict)
            and "phone" in slots
            and (
                session_state.active_request_kind == "checkout"
                or session_state.manager_handoff_stage == "awaiting_phone"
            )
        ):
            suggested = ["human_operator_request"] if session_state.manager_handoff_stage == "awaiting_phone" else ["legal_entity_purchase_flow"]
            return self._build(
                is_followup=True,
                followup_type="phone_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=suggested,
                reason="Phone-only reply detected in an active checkout or handoff flow.",
                session_state=session_state,
                user_query=query,
            )

        if (
            self._SHORT_PRICE_RE.match(query)
            and session_state.last_topic_ids
            and any(topic in session_state.last_topic_ids for topic in ("epc_tariffs", "tis_tariffs", "purchase_ready"))
        ):
            return self._build(
                is_followup=True,
                followup_type="pricing_followup",
                inherited_brand=inherited_brand,
                inherited_topics=inherited_topics,
                suggested_topics=["epc_tariffs", "tis_tariffs"],
                reason="Short pricing follow-up after a prior pricing or purchase turn.",
                session_state=session_state,
                user_query=query,
            )

        return self._build(
            is_followup=False,
            followup_type="unknown_followup",
            inherited_brand=inherited_brand,
            inherited_topics=inherited_topics,
            suggested_topics=[],
            reason="No follow-up rule matched.",
            session_state=session_state,
            user_query=query,
        )

    def _build(
        self,
        *,
        is_followup: bool,
        followup_type: str,
        inherited_brand: str,
        inherited_topics: list[str],
        suggested_topics: list[str],
        reason: str,
        session_state: SessionState,
        user_query: str,
    ) -> FollowupResolution:
        trace = {
            "is_followup": is_followup,
            "followup_type": followup_type,
            "input": user_query,
            "state_before": {
                "last_mentioned_brand": session_state.last_mentioned_brand,
                "last_topic_ids": list(session_state.last_topic_ids),
                "last_action_name": session_state.last_action_name,
                "last_bot_question_type": session_state.last_bot_question_type,
                "active_request_kind": session_state.active_request_kind,
                "manager_handoff_stage": session_state.manager_handoff_stage,
            },
            "inherited_brand": inherited_brand,
            "suggested_topics": list(suggested_topics),
            "reason": reason,
        }
        return FollowupResolution(
            is_followup=is_followup,
            followup_type=followup_type,
            inherited_brand=inherited_brand,
            inherited_topics=list(inherited_topics),
            suggested_topics=list(suggested_topics),
            reason=reason,
            trace=trace,
        )

    @staticmethod
    def _slot_matches_inherited_brand(*, slots: object, inherited_brand: str) -> bool:
        if not isinstance(slots, dict):
            return False
        normalized_inherited = str(inherited_brand or "").strip().lower()
        if not normalized_inherited:
            return False
        if str(slots.get("brand", "")).strip().lower() == normalized_inherited:
            return True
        for key in ("brands", "requested_brand_keys"):
            values = slots.get(key, [])
            if not isinstance(values, list):
                continue
            if any(str(item).strip().lower() == normalized_inherited for item in values):
                return True
        brand_mentions = slots.get("brand_mentions", [])
        if isinstance(brand_mentions, list):
            for item in brand_mentions:
                if not isinstance(item, dict):
                    continue
                if str(item.get("canonical_brand", "")).strip().lower() == normalized_inherited:
                    return True
                if str(item.get("normalized_key", "")).strip().lower() == normalized_inherited:
                    return True
        return False

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()
