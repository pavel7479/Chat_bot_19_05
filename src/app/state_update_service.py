from __future__ import annotations

from pathlib import Path

from src.app.pricing_flow_state_service import PricingFlowStateService
from src.core.models import BotResponse, SessionState, TopicClassificationResult
from src.core.turn_analysis import extract_turn_analysis


class StateUpdateService:
    """Apply deterministic post-response session state updates."""

    def __init__(self, brands_file_path: Path | None = None, pricing_flow_state_service: PricingFlowStateService | None = None) -> None:
        self._pricing_flow = pricing_flow_state_service or (
            PricingFlowStateService(brands_file_path) if brands_file_path is not None else None
        )

    def apply_after_classification(
        self,
        base_state: SessionState,
        topic_result: TopicClassificationResult,
    ) -> SessionState:
        merged_state_payload = base_state.as_dict()
        merged_state_payload.update(topic_result.state_snapshot)
        current_state = SessionState(**merged_state_payload)
        turn_analysis = extract_turn_analysis(topic_result)
        current_state.slots = {**current_state.slots, **dict(turn_analysis.slots)}
        current_state.last_focus_topic = topic_result.primary_topic_id
        current_state.last_bot_question_type = current_state.last_question_type
        detected_brands = turn_analysis.slots.get("brands", [])
        if not isinstance(detected_brands, list):
            detected_brands = []
        detected_brands = [str(item).strip().lower() for item in detected_brands if str(item).strip()]
        detected_brand = str(turn_analysis.slots.get("brand", "")).strip()
        if detected_brands:
            current_state.slots["brands"] = list(detected_brands)
        if detected_brand and len(detected_brands) <= 1:
            current_state.last_mentioned_brand = detected_brand
            current_state.slots["last_brand_source"] = str(
                turn_analysis.slots.get("last_brand_source", "query")
            ).strip() or "query"
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        dialog_act_trace = diagnostics.get("dialog_act_trace", {})
        if isinstance(dialog_act_trace, dict):
            extra_patch = dialog_act_trace.get("extra_state_patch", {})
            if isinstance(extra_patch, dict):
                for key, value in extra_patch.items():
                    if hasattr(current_state, key):
                        setattr(current_state, key, value)
        if topic_result.flow_name:
            current_state.active_flow = topic_result.flow_name
        if topic_result.flow_step:
            current_state.flow_step = topic_result.flow_step
        if current_state.active_flow == "company_documents":
            current_state.active_request_kind = "documents"
        elif current_state.active_flow in {"checkout", "pricing", "post_payment_support"}:
            current_state.active_request_kind = current_state.active_flow
        if self._pricing_flow is not None:
            pricing_patch = self._pricing_flow.apply_classification_event(
                session_state=current_state,
                topic_result=topic_result,
            ).to_session_patch()
            for key, value in pricing_patch.items():
                if hasattr(current_state, key):
                    setattr(current_state, key, value)
        current_state.active_business_flow = self._infer_business_flow_after_classification(current_state)
        return current_state

    def apply_after_response(
        self,
        state_before_response: SessionState,
        previous_state: SessionState,
        response: BotResponse,
        answer_block: dict[str, object] | None = None,
    ) -> SessionState:
        current_state = SessionState(**state_before_response.as_dict())
        if response.action_name:
            if response.action_name == previous_state.last_action_name:
                current_state.same_action_repeats = previous_state.same_action_repeats + 1
            else:
                current_state.same_action_repeats = 1
                current_state.last_action_name = response.action_name

        if response.action_name in {"human_operator", "human_operator_collect_phone"}:
            current_state.manager_handoff_stage = "awaiting_phone"
        elif response.action_name == "human_operator_phone_invalid":
            current_state.manager_handoff_stage = "phone_invalid"
        elif response.action_name == "human_operator_phone_confirm":
            current_state.manager_handoff_stage = "confirmed"

        if response.action_name == "company_services":
            semantic_memory = dict(current_state.service_semantic_memory) if isinstance(current_state.service_semantic_memory, dict) else {}
            used_service_fact_ids = semantic_memory.get("used_fact_ids", [])
            if not isinstance(used_service_fact_ids, list):
                used_service_fact_ids = []
            used_service_groups = semantic_memory.get("used_groups", [])
            if not isinstance(used_service_groups, list):
                used_service_groups = []
            for fact_id in response.used_evidence_ids:
                normalized = str(fact_id).strip()
                if normalized and (normalized.startswith("company_services") or normalized == "catalog_list_products") and normalized not in used_service_fact_ids:
                    used_service_fact_ids.append(normalized)
            if isinstance(answer_block, dict):
                for semantic_group in answer_block.get("used_service_semantic_groups", []):
                    normalized_group = str(semantic_group).strip()
                    if normalized_group and normalized_group not in used_service_groups:
                        used_service_groups.append(normalized_group)
            current_state.service_semantic_memory = {
                "used_fact_ids": used_service_fact_ids,
                "used_groups": used_service_groups,
            }
            current_state.slots["used_service_fact_ids"] = list(used_service_fact_ids)
            current_state.slots["used_service_semantic_groups"] = list(used_service_groups)
            if "company_services_documents" in response.used_evidence_ids or "company_documents_contact_followup" in response.used_evidence_ids:
                current_state.active_request_kind = "documents"
                current_state.active_flow = "company_documents"
                current_state.flow_step = "clarify_documents"
        elif response.action_name in {"request_requisites", "ask_legal_status", "manager_handoff_ready"}:
            current_state.active_request_kind = "checkout"
        elif response.action_name in {"post_payment_no_access_handoff", "post_payment_access_info"}:
            current_state.active_request_kind = "post_payment_support"

        if response.action_name == "company_services" and "company_documents_contact_followup" in response.used_evidence_ids:
            current_state.document_contact_collected = True

        current_state.last_bot_question_type = self._infer_last_bot_question_type(response)

        if "evidence=" in response.reasoning_summary:
            current_state.evidence_status = response.reasoning_summary.split("evidence=")[-1].split(";")[0]
        if self._pricing_flow is not None:
            pricing_patch = self._pricing_flow.apply_response_event(
                state_before_response=state_before_response,
                response_action=response.action_name,
                answer_block=answer_block,
            ).to_session_patch()
            for key, value in pricing_patch.items():
                if hasattr(current_state, key):
                    setattr(current_state, key, value)
        current_state.active_business_flow = self._infer_business_flow_after_response(
            current_state=current_state,
            response=response,
        )
        return current_state

    @staticmethod
    def _infer_business_flow_after_classification(current_state: SessionState) -> str:
        active_request = str(current_state.active_request_kind or "").strip()
        if active_request in {"company_services", "documents"}:
            return "company_services"
        if current_state.active_pricing_flow == "tis":
            return "pricing_tis"
        if active_request == "checkout":
            return "purchase_flow"
        if active_request == "human_operator":
            return "manager_handoff"
        return str(current_state.active_business_flow or "none").strip() or "none"

    @staticmethod
    def _infer_business_flow_after_response(
        *,
        current_state: SessionState,
        response: BotResponse,
    ) -> str:
        action = str(response.action_name or "").strip()
        if action in {"greeting_once", "company_services"}:
            return "company_services"
        if action in {"pricing_summary", "tis_tariffs", "epc_tariffs"}:
            return "pricing_tis" if current_state.active_pricing_flow == "tis" else "pricing_epc"
        if action in {"ask_legal_status", "request_requisites", "manager_handoff_ready"}:
            return "purchase_flow"
        if action in {"human_operator", "human_operator_collect_phone", "human_operator_phone_invalid", "human_operator_phone_confirm"}:
            return "manager_handoff"
        return str(current_state.active_business_flow or "none").strip() or "none"

    @staticmethod
    def _infer_last_bot_question_type(response: BotResponse) -> str:
        if response.action_name == "ask_legal_status":
            return "legal_status_check"
        if response.action_name == "request_requisites":
            return "purchase_confirmation"
        if response.action_name == "manager_handoff_ready":
            return "purchase_confirmation"
        if response.action_name == "tis_tariffs" and "?" in response.answer_text:
            return "pricing_clarification"
        if response.action_name == "brand_availability" and "?" in response.answer_text:
            return "brand_clarification"
        if response.action_name in {"epc_tariffs", "tis_tariffs", "compare_epc_tis"} and "?" in response.answer_text:
            return "pricing_clarification"
        if response.action_name == "company_services" and "?" in response.answer_text:
            return "service_clarification"
        return "unknown"
