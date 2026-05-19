from __future__ import annotations

import re
from typing import Any

from src.topics.dto import TurnSignals


class TurnSignalService:
    """Builds turn-level semantic DTO for planner/retrieval."""

    def build(
        self,
        context: Any,
        signals: set[str],
        normalized_topics: list[str],
        locked_topics: set[str],
        slots: dict[str, bool],
    ) -> TurnSignals:
        query = context.query_lower
        nonsense_input = bool(slots.get("nonsense_input", False))
        abuse_message = bool(slots.get("abuse_message", False))
        out_of_scope_current_query = bool(
            slots.get("geo_map_request", False)
            or slots.get("out_of_scope_catalog", False)
            or re.search(r"\b(велосипед\w*|самокат\w*|космическ\w*|корабл\w*|дрова|одежд\w*|редк\w*\s+пород\w*\s+рыб)\b", query)
        )
        current_focus = "unknown"
        if nonsense_input or abuse_message:
            current_focus = "clarify_request"
        elif out_of_scope_current_query:
            current_focus = "out_of_scope_response"
        else:
            for candidate in (
                "out_of_scope_response",
                "parts_selection_not_supported",
                "company_services_documents",
                "macos_support",
                "compare_epc_tis",
                "tis_tariffs",
                "epc_tariffs",
                "brand_availability",
                "api_integration",
                "human_operator_request",
                "demo_policy",
                "payment_flow",
                "company_services",
            ):
                if candidate in locked_topics:
                    current_focus = candidate
                    break

        docs_query = bool(
            slots.get("company_documents_card", False)
            or re.search(r"\b(договор|карточк\w*|документ\w*)\b", query)
            or (
                re.search(r"\b(договор|карточк\w*|документ\w*)\b", context.history_lower)
                and re.search(r"\b(ип|юр|юрид|инн|телефон|контакт|номер|\+?\d{5,})\b", query)
            )
        )
        pricing_request = bool(slots.get("pricing_request", False))
        catalog_list_request = bool(slots.get("catalog_list_request", False))
        feature_comparison = bool(slots.get("feature_comparison", False))
        refund_policy = bool(slots.get("refund_policy", False))
        post_payment_access = bool(slots.get("post_payment_access", False))
        multi_user_access = bool(slots.get("multi_user_access", False))
        services_query = bool(
            slots.get("company_services_overview", False)
            or re.search(
                r"(чем ты мне можешь помочь|что ты умеешь|подскажи|какой каталог лучше|чем вы лучше|скидк\w*|что еще умеет|что еще можете)",
                query,
            )
        )
        out_of_scope = bool(
            out_of_scope_current_query
            or "out_of_scope_catalog" in signals
            or ("out_of_scope_request" in normalized_topics and "domain" not in signals)
        )
        parts_query = "parts_selection" in signals
        manager_query = "human" in signals or "human_operator_request" in normalized_topics
        payment_detail_query = bool(
            re.search(r"\b(ип|юр|юрид|инн|реквиз|сч[её]т|счет|qr|оплат|телефон|контакт)\w*\b", query)
        )

        if current_focus == "unknown":
            if out_of_scope:
                current_focus = "out_of_scope_response"
            elif parts_query:
                current_focus = "parts_selection_not_supported"
            elif docs_query:
                current_focus = "company_services_documents"
            elif "macos_support" in normalized_topics:
                current_focus = "macos_support"
            elif "compare" in query and "epc" in query and "tis" in query:
                current_focus = "compare_epc_tis"
            elif "tis_tariffs" in normalized_topics and "epc_tariffs" in normalized_topics:
                current_focus = "compare_epc_tis"
            elif "tis_tariffs" in normalized_topics:
                current_focus = "tis_tariffs"
            elif "epc_tariffs" in normalized_topics:
                current_focus = "epc_tariffs"
            elif "brand_list_request" in normalized_topics or "specific_brand_check" in normalized_topics:
                current_focus = "brand_availability"
            elif "demo_access" in normalized_topics:
                current_focus = "demo_policy"
            elif "payment_without_details" in normalized_topics or "legal_entity_purchase_flow" in normalized_topics:
                current_focus = "payment_flow"
            elif "purchase_ready" in normalized_topics:
                current_focus = "payment_flow" if payment_detail_query else "purchase_entry"
            elif manager_query:
                current_focus = "human_operator_request"
            elif "api_integration" in normalized_topics:
                current_focus = "api_integration"
            elif feature_comparison and ("epc_tariffs" in normalized_topics or "tis_tariffs" in normalized_topics):
                current_focus = "compare_epc_tis"
            elif pricing_request and "tis_tariffs" in normalized_topics:
                current_focus = "tis_tariffs"
            elif pricing_request and "epc_tariffs" in normalized_topics:
                current_focus = "epc_tariffs"
            elif catalog_list_request:
                current_focus = "brand_availability"
            elif services_query or "company_services_info" in normalized_topics:
                current_focus = "company_services"
        if current_focus == "unknown":
            for topic in normalized_topics:
                if topic and topic not in {"out_of_scope_request"}:
                    current_focus = topic
                    break
        if current_focus == "unknown":
            current_focus = "clarify_request"

        return TurnSignals(
            signals=sorted(signals),
            locked_topics=sorted(locked_topics),
            slots=slots,
            current_focus=current_focus,
            docs_query=docs_query,
            services_overview_query=services_query,
            out_of_scope_query=out_of_scope,
            out_of_scope_current_query=out_of_scope_current_query,
            parts_query=parts_query,
            manager_query=manager_query,
            nonsense_input=nonsense_input,
            abuse_message=abuse_message,
            pricing_request=pricing_request,
            catalog_list_request=catalog_list_request,
            feature_comparison=feature_comparison,
            refund_policy=refund_policy,
            post_payment_access=post_payment_access,
            multi_user_access=multi_user_access,
        )
