from __future__ import annotations

from src.agents.response_planner import ResponsePlan
from src.agents.policy.types import ResponseAction
from src.agents.response_services.meaning_validator import MeaningValidationResult
from src.app.price_provider import PriceContext
from src.app.pricing_meaning_policy import PricingMeaningPolicy
from src.core.models import MandatoryMeaningBlock, PreparedResponseContext


class MeaningOwnershipPolicy:
    def __init__(self, pricing_policy: PricingMeaningPolicy | None = None) -> None:
        self._pricing_policy = pricing_policy or PricingMeaningPolicy()

    def build(
        self,
        *,
        selected_action: ResponseAction,
        response_plan: ResponsePlan,
        price_context: PriceContext,
        structured_context: PreparedResponseContext,
    ) -> list[MandatoryMeaningBlock]:
        blocks: list[MandatoryMeaningBlock] = []
        if selected_action.name in {"tis_tariffs", "pricing_summary", "epc_tariffs"} or response_plan.required_price_blocks:
            blocks.extend(self._pricing_policy.build_blocks(price_context))
        brand_display = str(structured_context.slots.get("brand_display", "")).strip()
        if brand_display and selected_action.name in {"brand_availability", "tis_tariffs"}:
            blocks.append(
                MandatoryMeaningBlock(
                    key="brand_display",
                    required_phrases=[brand_display],
                    semantic_tags=[],
                )
            )
        if selected_action.name == "human_operator":
            blocks.append(
                MandatoryMeaningBlock(
                    key="human_operator_handoff",
                    required_phrases=["менеджер"],
                    semantic_tags=["human_operator_handoff"],
                )
            )
        if selected_action.name == "brand_group_clarification":
            blocks.append(
                MandatoryMeaningBlock(
                    key="brand_group_clarification",
                    required_phrases=["VAG", "конкретную марку"],
                    semantic_tags=[],
                )
            )
        if selected_action.name == "partial_catalog_restriction":
            blocks.append(
                MandatoryMeaningBlock(
                    key="partial_catalog_restriction",
                    required_phrases=["EPC Full", "полным пакетом", "TIS"],
                    semantic_tags=[],
                )
            )
        if selected_action.name == "ask_legal_status":
            blocks.append(
                MandatoryMeaningBlock(
                    key="ask_legal_status",
                    required_phrases=["юрлицо", "ИП"],
                    semantic_tags=[],
                )
            )
        return blocks
