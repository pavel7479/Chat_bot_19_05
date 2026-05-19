from __future__ import annotations

from dataclasses import dataclass, field

from src.core.models import PricingBrandItem, PricingBrandStatus, PricingFlowMode, PricingFlowState
from src.domain.pricing import PriceCatalog


@dataclass(slots=True)
class PriceContext:
    product: str = ""
    brands: list[str] = field(default_factory=list)
    recognized_brands: list[str] = field(default_factory=list)
    unknown_brand_mentions: list[str] = field(default_factory=list)
    priced_brands: list[str] = field(default_factory=list)
    price_lines: list[str] = field(default_factory=list)
    evidence_items: list[dict[str, object]] = field(default_factory=list)
    missing_tis_price_brands: list[str] = field(default_factory=list)
    rendered_priced_brands: list[str] = field(default_factory=list)
    rendered_missing_tis_price_brands: list[str] = field(default_factory=list)
    rendered_unresolved_surfaces: list[str] = field(default_factory=list)
    tis_price_status: str = "not_requested"
    fallback_price_blocks: list[str] = field(default_factory=list)
    pricing_mode: str = "all"

    @property
    def lines(self) -> list[str]:
        return list(self.price_lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "product": self.product,
            "brands": list(self.brands),
            "recognized_brands": list(self.recognized_brands),
            "unknown_brand_mentions": list(self.unknown_brand_mentions),
            "priced_brands": list(self.priced_brands),
            "price_lines": list(self.price_lines),
            "evidence_items": list(self.evidence_items),
            "missing_tis_price_brands": list(self.missing_tis_price_brands),
            "rendered_priced_brands": list(self.rendered_priced_brands),
            "rendered_missing_tis_price_brands": list(self.rendered_missing_tis_price_brands),
            "rendered_unresolved_surfaces": list(self.rendered_unresolved_surfaces),
            "tis_price_status": self.tis_price_status,
            "fallback_price_blocks": list(self.fallback_price_blocks),
            "pricing_mode": self.pricing_mode,
        }


class PriceProvider:
    def __init__(self, price_catalog: PriceCatalog | None = None) -> None:
        self._price_catalog = price_catalog

    def build(
        self,
        required_price_blocks: list[str],
        normalized_brands: list[object],
        followup_trace: dict[str, object] | None = None,
        raw_brand_mentions: list[object] | None = None,
        unknown_brand_mentions: list[object] | None = None,
        pricing_mode: str = "all",
    ) -> PriceContext:
        recognized = [str(item).strip().lower() for item in normalized_brands if str(item).strip()]
        raw_mentions = [str(item).strip() for item in (raw_brand_mentions or []) if str(item).strip()]
        unknown_mentions = [str(item).strip() for item in (unknown_brand_mentions or []) if str(item).strip()]
        brand_items: list[PricingBrandItem] = [
            PricingBrandItem(
                canonical_brand=canonical,
                display_name=self._display_surface(canonical),
                raw_surface=canonical,
                recognized=True,
                status=PricingBrandStatus.MISSING_PRICE,
            )
            for canonical in recognized
        ]
        brand_items.extend(
            PricingBrandItem(
                canonical_brand="",
                display_name=self._display_surface(surface),
                raw_surface=surface,
                recognized=False,
                status=PricingBrandStatus.UNRESOLVED,
            )
            for surface in unknown_mentions
        )
        mode_value = str(pricing_mode or PricingFlowMode.ALL.value).strip() or PricingFlowMode.ALL.value
        try:
            mode = PricingFlowMode(mode_value)
        except ValueError:
            mode = PricingFlowMode.ALL
        pricing_flow = PricingFlowState(
            active=bool(normalized_brands or raw_brand_mentions or unknown_brand_mentions),
            product="tis" if "tis" in required_price_blocks else ("epc" if "epc" in required_price_blocks else ""),
            mode=mode,
            brand_items=brand_items,
            raw_brand_mentions=raw_mentions,
            requested_brand_keys=[str(item).strip().lower() for item in raw_mentions if str(item).strip()],
            recognized_brands=recognized,
            unknown_brand_mentions=unknown_mentions,
        )
        return self.build_for_pricing_flow(
            required_price_blocks=required_price_blocks,
            pricing_flow=pricing_flow,
            followup_trace=followup_trace if isinstance(followup_trace, dict) else {},
        )

    def build_for_pricing_flow(
        self,
        *,
        required_price_blocks: list[str],
        pricing_flow: PricingFlowState,
        followup_trace: dict[str, object] | None = None,
    ) -> PriceContext:
        blocks = [str(block).strip() for block in required_price_blocks if str(block).strip()]
        brand_items = [self._clone_item(item) for item in pricing_flow.brand_items]
        known_brands = [
            str(item.canonical_brand).strip().lower()
            for item in brand_items
            if item.recognized and str(item.canonical_brand).strip()
        ] or [str(item).strip().lower() for item in pricing_flow.recognized_brands if str(item).strip()]
        unknown_surfaces = [
            str(item.display_name or item.raw_surface).strip()
            for item in brand_items
            if not item.recognized and str(item.display_name or item.raw_surface).strip()
        ] or [str(item).strip() for item in pricing_flow.unknown_brand_mentions if str(item).strip()]
        evidence_items: list[dict[str, object]] = []
        price_lines: list[str] = []
        priced_brands: list[str] = []
        missing_tis_price_brands: list[str] = []
        rendered_priced_brands: list[str] = []
        rendered_missing_tis_price_brands: list[str] = []
        rendered_unresolved_surfaces: list[str] = []
        fallback_price_blocks: list[str] = []
        product = "mixed"
        tis_price_status = "not_requested"
        pricing_mode = pricing_flow.mode.value if isinstance(pricing_flow.mode, PricingFlowMode) else str(pricing_flow.mode or "all").strip() or "all"
        trace = followup_trace if isinstance(followup_trace, dict) else {}

        if self._price_catalog is None:
            return PriceContext(
                product=product,
                brands=known_brands,
                recognized_brands=known_brands,
                unknown_brand_mentions=unknown_surfaces,
                priced_brands=priced_brands,
                price_lines=price_lines,
                evidence_items=evidence_items,
                missing_tis_price_brands=missing_tis_price_brands,
                rendered_priced_brands=rendered_priced_brands,
                rendered_missing_tis_price_brands=rendered_missing_tis_price_brands,
                rendered_unresolved_surfaces=rendered_unresolved_surfaces,
                tis_price_status=tis_price_status,
                fallback_price_blocks=fallback_price_blocks,
                pricing_mode=pricing_mode,
            )

        if "epc" in blocks:
            epc_text = self._format_epc_prices()
            if epc_text:
                product = "epc" if "tis" not in blocks else "mixed"
                price_lines.append(epc_text)
                evidence_items.append(self._evidence("epc_prices", epc_text, 10.0))

        if "tis" not in blocks:
            return PriceContext(
                product=product,
                brands=known_brands,
                recognized_brands=known_brands,
                unknown_brand_mentions=unknown_surfaces,
                priced_brands=priced_brands,
                price_lines=price_lines,
                evidence_items=evidence_items,
                missing_tis_price_brands=missing_tis_price_brands,
                rendered_priced_brands=rendered_priced_brands,
                rendered_missing_tis_price_brands=rendered_missing_tis_price_brands,
                rendered_unresolved_surfaces=rendered_unresolved_surfaces,
                tis_price_status=tis_price_status,
                fallback_price_blocks=fallback_price_blocks,
                pricing_mode=pricing_mode,
            )

        product = "tis" if "epc" not in blocks else "mixed"
        priced_items = self._select_priced_items(brand_items)
        missing_items = self._select_missing_price_items(brand_items)
        unresolved_items = self._select_unresolved_items(brand_items)
        priced_brands = [str(item.canonical_brand).strip().lower() for item in priced_items]
        missing_tis_price_brands = [str(item.canonical_brand).strip().lower() for item in missing_items]

        if pricing_mode == PricingFlowMode.SEPARATE.value:
            tis_price_status = self._status_for(priced_brands, missing_tis_price_brands, unknown_surfaces)
            for item in priced_items:
                canonical = str(item.canonical_brand).strip().lower()
                display_name = self._price_catalog.display_name(canonical)
                amount = self._price_catalog.tis_price_for(canonical)
                line = f"TIS для {display_name} — {amount} руб."
                price_lines.append(line)
                evidence_items.append(self._evidence(f"tis_price:{canonical}", line, 10.0))
                rendered_priced_brands.append(canonical)
            for item in missing_items:
                canonical = str(item.canonical_brand).strip().lower()
                display_name = self._price_catalog.display_name(canonical)
                line = f"По {display_name} цена TIS в текущем прайсе не указана."
                price_lines.append(line)
                evidence_items.append(self._evidence(f"tis_price:missing:{canonical}", line, 9.0))
                rendered_missing_tis_price_brands.append(canonical)
            for item in unresolved_items:
                line = f"{self._display_surface(item.display_name or item.raw_surface)} не распознан как бренд."
                price_lines.append(line)
                rendered_unresolved_surfaces.append(self._display_surface(item.display_name or item.raw_surface))
            if unknown_surfaces:
                evidence_items.append(self._evidence("tis_price:unknown_brands", "\n".join(price_lines), 8.8))
            if (missing_tis_price_brands or unknown_surfaces) and "epc" not in blocks:
                self._append_epc_fallback(price_lines, evidence_items, fallback_price_blocks)
        elif pricing_mode == PricingFlowMode.REMAINING_ONLY.value:
            remaining_items = self._select_remaining_items(brand_items)
            remaining_missing_items = self._select_missing_price_items(remaining_items)
            remaining_unknown_items = self._select_unresolved_items(remaining_items)
            remaining_missing = [str(item.canonical_brand).strip().lower() for item in remaining_missing_items]
            remaining_unknown = [str(item.display_name or item.raw_surface).strip() for item in remaining_unknown_items]
            tis_price_status = self._status_for([], remaining_missing, remaining_unknown)
            for item in remaining_missing_items:
                canonical = str(item.canonical_brand).strip().lower()
                display_name = self._price_catalog.display_name(canonical)
                line = f"По {display_name} цена TIS в текущем прайсе не указана."
                price_lines.append(line)
                evidence_items.append(self._evidence(f"tis_price:missing:{canonical}", line, 9.0))
                rendered_missing_tis_price_brands.append(canonical)
            if remaining_unknown:
                unknown_line = (
                    f"Не распознал как бренды: {', '.join(self._display_surface(item) for item in remaining_unknown)}. "
                    "Уточните конкретные марки, и я проверю их отдельно."
                )
                price_lines.append(unknown_line)
                evidence_items.append(self._evidence("tis_price:unknown_brands", unknown_line, 8.8))
                rendered_unresolved_surfaces.extend([self._display_surface(item) for item in remaining_unknown])
            if (remaining_missing or remaining_unknown) and "epc" not in blocks:
                self._append_epc_fallback(price_lines, evidence_items, fallback_price_blocks)
        elif pricing_mode == PricingFlowMode.EXPLAIN_UNRESOLVED.value:
            remaining_items = self._select_remaining_items(brand_items)
            explain_missing_items = self._select_missing_price_items(remaining_items)
            unresolved_items = self._select_unresolved_items(remaining_items)
            explain_missing = [str(item.canonical_brand).strip().lower() for item in explain_missing_items]
            unresolved_surfaces = [str(item.display_name or item.raw_surface).strip() for item in unresolved_items]
            tis_price_status = self._status_for([], explain_missing, unresolved_surfaces)
            for item in explain_missing_items:
                canonical = str(item.canonical_brand).strip().lower()
                display_name = self._price_catalog.display_name(canonical)
                line = f"По {display_name} цена TIS в текущем прайсе не указана."
                price_lines.append(line)
                evidence_items.append(self._evidence(f"tis_price:missing:{canonical}", line, 9.0))
                rendered_missing_tis_price_brands.append(canonical)
            unknown_lines: list[str] = []
            for item in unresolved_items:
                display_surface = self._display_surface(item.display_name or item.raw_surface)
                if display_surface.lower() == "captiva":
                    line = f"{display_surface} похожа на модель, нужна конкретная марка."
                else:
                    line = f"{display_surface} не распознан как бренд. Уточните конкретную марку, и я проверю её отдельно."
                price_lines.append(line)
                unknown_lines.append(line)
                rendered_unresolved_surfaces.append(display_surface)
            if unknown_lines:
                evidence_items.append(self._evidence("tis_price:unknown_brands", "\n".join(unknown_lines), 8.8))
        elif priced_items or missing_items or unresolved_items:
            found_lines, _ = self._format_tis_prices(priced_brands)
            tis_price_status = self._status_for(priced_brands, missing_tis_price_brands, unknown_surfaces)
            for line, brand in found_lines:
                price_lines.append(line)
                evidence_items.append(self._evidence(f"tis_price:{brand}", line, 10.0))
                rendered_priced_brands.append(brand)
            for canonical in missing_tis_price_brands:
                display_name = self._price_catalog.display_name(canonical)
                line = f"По {display_name} цена TIS в текущем прайсе не указана."
                price_lines.append(line)
                evidence_items.append(self._evidence(f"tis_price:missing:{canonical}", line, 9.0))
                rendered_missing_tis_price_brands.append(canonical)
            if unknown_surfaces:
                unknown_line = (
                    f"Не распознал как бренды: {', '.join(self._display_surface(item) for item in unknown_surfaces)}. "
                    "Уточните конкретные марки, и я проверю их отдельно."
                )
                price_lines.append(unknown_line)
                evidence_items.append(self._evidence("tis_price:unknown_brands", unknown_line, 8.8))
                rendered_unresolved_surfaces.extend([self._display_surface(item) for item in unknown_surfaces])
            if (missing_tis_price_brands or unknown_surfaces) and "epc" not in blocks:
                self._append_epc_fallback(price_lines, evidence_items, fallback_price_blocks)
        else:
            if known_brands:
                tis_price_status = "missing"
                missing_tis_price_brands = list(known_brands)
                brand_names = ", ".join(self._price_catalog.display_name(brand) for brand in known_brands)
                message = f"Для TIS по бренду {brand_names} цена в текущем прайсе не указана."
                price_lines.append(message)
                evidence_items.append(self._evidence("tis_price:price_not_found", message, 9.0))
                rendered_missing_tis_price_brands.extend(missing_tis_price_brands)
                self._append_epc_fallback(price_lines, evidence_items, fallback_price_blocks)
            elif pricing_flow.raw_brand_mentions:
                message = "Не удалось распознать бренды из списка. Напишите конкретные марки, и я проверю их по TIS отдельно."
                price_lines.append(message)
                evidence_items.append(self._evidence("tis_price:brand_unrecognized", message, 9.0))
                rendered_unresolved_surfaces.extend(unknown_surfaces)
            else:
                if trace.get("is_followup") and str(trace.get("followup_type", "")).strip() == "brand_price_followup":
                    message = "Для TIS цена зависит от бренда, но бренд в follow-up не удалось надёжно восстановить. Как только вы подтвердите марку, сразу покажу цену."
                else:
                    message = "Для расчета стоимости TIS нужна конкретная марка. Напишите интересующие бренды, и я сразу покажу цену."
                price_lines.append(message)
                evidence_items.append(self._evidence("tis_price:brand_required", message, 9.0))

        return PriceContext(
            product=product,
            brands=known_brands,
            recognized_brands=known_brands,
            unknown_brand_mentions=unknown_surfaces,
            priced_brands=priced_brands,
            price_lines=price_lines,
            evidence_items=evidence_items,
            missing_tis_price_brands=missing_tis_price_brands,
            rendered_priced_brands=rendered_priced_brands,
            rendered_missing_tis_price_brands=rendered_missing_tis_price_brands,
            rendered_unresolved_surfaces=rendered_unresolved_surfaces,
            tis_price_status=tis_price_status,
            fallback_price_blocks=fallback_price_blocks,
            pricing_mode=pricing_mode,
        )

    def _select_priced_items(self, items: list[PricingBrandItem]) -> list[PricingBrandItem]:
        if self._price_catalog is None:
            return []
        return [
            item
            for item in items
            if item.recognized
            and str(item.canonical_brand).strip()
            and self._price_catalog.has_tis_price(str(item.canonical_brand).strip().lower())
        ]

    def _select_missing_price_items(self, items: list[PricingBrandItem]) -> list[PricingBrandItem]:
        if self._price_catalog is None:
            return []
        return [
            item
            for item in items
            if item.recognized
            and str(item.canonical_brand).strip()
            and not self._price_catalog.has_tis_price(str(item.canonical_brand).strip().lower())
        ]

    @staticmethod
    def _select_unresolved_items(items: list[PricingBrandItem]) -> list[PricingBrandItem]:
        return [item for item in items if not item.recognized]

    @staticmethod
    def _select_remaining_items(items: list[PricingBrandItem]) -> list[PricingBrandItem]:
        return [item for item in items if not item.processed]

    @staticmethod
    def _clone_item(item: PricingBrandItem) -> PricingBrandItem:
        status = item.status if isinstance(item.status, PricingBrandStatus) else PricingBrandStatus.UNRESOLVED
        return PricingBrandItem(
            canonical_brand=str(item.canonical_brand).strip().lower(),
            display_name=str(item.display_name).strip(),
            raw_surface=str(item.raw_surface).strip(),
            recognized=bool(item.recognized),
            has_price=bool(item.has_price),
            processed=bool(item.processed),
            status=status,
        )

    def _format_epc_prices(self) -> str:
        order = [
            ("1_month", "1 месяц"),
            ("3_months", "3 месяца"),
            ("6_months", "6 месяцев"),
            ("12_months", "12 месяцев"),
        ]
        items: list[str] = []
        for key, label in order:
            if not self._price_catalog.has_epc_period(key):
                continue
            items.append(f"{label} — {self._price_catalog.epc_price_for(key)} руб")
        if not items:
            return ""
        return "Тариф EPC Full: " + ", ".join(items) + "."

    def _format_tis_prices(self, normalized_brands: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
        lines: list[tuple[str, str]] = []
        missing: list[str] = []
        seen: set[str] = set()
        for brand in normalized_brands:
            canonical = str(brand).strip().lower()
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            if not self._price_catalog.has_tis_price(canonical):
                missing.append(canonical)
                continue
            display_name = self._price_catalog.display_name(canonical)
            amount = self._price_catalog.tis_price_for(canonical)
            lines.append((f"TIS для {display_name} — {amount} руб.", canonical))
        return lines, missing

    def _append_epc_fallback(
        self,
        price_lines: list[str],
        evidence_items: list[dict[str, object]],
        fallback_price_blocks: list[str],
    ) -> None:
        epc_text = self._format_epc_prices()
        if epc_text and epc_text not in price_lines:
            price_lines.append(epc_text)
            fallback_price_blocks.append("epc")
            evidence_items.append(self._evidence("epc_prices:fallback", epc_text, 9.5))

    @staticmethod
    def _status_for(priced_brands: list[str], missing_brands: list[str], unknown_brands: list[str]) -> str:
        if missing_brands and not priced_brands and not unknown_brands:
            return "missing"
        if priced_brands and not missing_brands and not unknown_brands:
            return "found"
        if priced_brands or missing_brands or unknown_brands:
            return "mixed"
        return "not_requested"

    @staticmethod
    def _evidence(evidence_id: str, text: str, score: float) -> dict[str, object]:
        return {
            "evidence_id": evidence_id,
            "text": text,
            "score": score,
            "source": "prices.yaml",
            "section_tag": "pricing",
        }

    @staticmethod
    def _display_surface(value: object) -> str:
        surface = str(value).strip()
        if not surface:
            return ""
        if surface.islower() and surface.isascii():
            return surface.capitalize()
        return surface
