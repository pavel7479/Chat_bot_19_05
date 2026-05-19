from __future__ import annotations

from src.app.price_provider import PriceContext
from src.core.models import MandatoryMeaningBlock


class PricingMeaningPolicy:
    def build_blocks(self, price_context: PriceContext) -> list[MandatoryMeaningBlock]:
        if str(price_context.tis_price_status).strip() == "missing":
            return [
                MandatoryMeaningBlock(
                    key="tis_missing",
                    required_phrases=["цена TIS", "не указана"],
                    semantic_tags=["tis_price_missing"],
                ),
                MandatoryMeaningBlock(
                    key="epc_fallback",
                    required_phrases=["EPC Full"],
                    semantic_tags=["epc_fallback_available"],
                ),
            ]
        return []
