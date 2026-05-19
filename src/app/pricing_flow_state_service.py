from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.models import (
    BrandMention,
    PricingBrandItem,
    PricingBrandStatus,
    PricingFlowMode,
    PricingFlowState,
    SessionState,
    TopicClassificationResult,
)
from src.domain.brands import BrandAliasResolver


class PricingFlowEvent:
    BRAND_LIST_RECEIVED = "brand_list_received"
    SEPARATE_REQUESTED = "separate_requested"
    REMAINING_REQUESTED = "remaining_requested"
    CLARIFICATION_PUSHBACK = "clarification_pushback"
    PRICE_RESULT_APPLIED = "price_result_applied"
    HUMAN_HANDOFF = "human_handoff"


@dataclass(slots=True)
class PricingFlowPatch:
    flow: PricingFlowState

    def to_session_patch(self) -> dict[str, object]:
        derived = PricingFlowStateService._sync_derived_fields(self.flow)
        return {
            "active_pricing_flow": derived.product if derived.active else "none",
            "pricing_requested_product": derived.product if derived.active else "",
            "requested_brands": list(derived.requested_brand_keys),
            "recognized_brands": list(derived.recognized_brands),
            "unknown_brand_mentions": list(derived.unknown_brand_mentions),
            "missing_price_brands": list(derived.missing_price_brands),
            "priced_brands": list(derived.priced_brands),
            "pending_brand_mentions": list(derived.pending_brand_mentions),
            "pricing_mode": derived.mode.value,
            "pricing_flow": derived.as_dict(),
        }


class PricingFlowStateService:
    def __init__(self, brands_file_path: Path) -> None:
        self._brand_resolver = BrandAliasResolver(brands_file_path)

    def from_session_state(self, session_state: SessionState) -> PricingFlowState:
        raw = session_state.pricing_flow if isinstance(session_state.pricing_flow, dict) else {}
        if raw:
            flow = PricingFlowState(
                active=bool(raw.get("active", False)),
                product=str(raw.get("product", "")).strip(),
                mode=self._parse_mode(raw.get("mode", PricingFlowMode.ALL.value)),
                stage=str(raw.get("stage", "idle")).strip() or "idle",
            )
            flow.brand_mentions = self._parse_brand_mentions(raw.get("brand_mentions", []))
            flow.brand_items = self._parse_brand_items(raw.get("brand_items", []), flow.brand_mentions)
            flow.raw_brand_mentions = self._string_list(raw.get("raw_brand_mentions", []))
            flow.requested_brand_keys = self._string_list(raw.get("requested_brand_keys", []))
            return self._sync_derived_fields(flow)

        product = str(session_state.pricing_requested_product or session_state.active_pricing_flow or "").strip().lower()
        active = bool(product and product != "none")
        flow = PricingFlowState(
            active=active,
            product=product if active else "",
            mode=self._parse_mode(session_state.pricing_mode or PricingFlowMode.ALL.value),
            stage="collect_brands" if active else "idle",
        )
        recognized = self._normalized_list(session_state.recognized_brands)
        unknown = self._string_list(session_state.unknown_brand_mentions)
        requested_keys = self._string_list(session_state.requested_brands)
        flow.brand_items = self._build_brand_items_from_compatibility(
            requested_brand_keys=requested_keys,
            recognized_brands=recognized,
            unknown_brand_mentions=unknown,
            priced_brands=self._normalized_list(session_state.priced_brands),
            missing_price_brands=self._normalized_list(session_state.missing_price_brands),
        )
        return self._sync_derived_fields(flow)

    def apply_classification_event(
        self,
        *,
        session_state: SessionState,
        topic_result: TopicClassificationResult,
    ) -> PricingFlowPatch:
        flow = self.from_session_state(session_state)
        diagnostics = topic_result.diagnostics if isinstance(topic_result.diagnostics, dict) else {}
        slot_trace = diagnostics.get("slot_extraction_trace", {}) if isinstance(diagnostics.get("slot_extraction_trace", {}), dict) else {}
        slots = slot_trace.get("slots", {}) if isinstance(slot_trace.get("slots", {}), dict) else {}
        dialog_act_trace = diagnostics.get("dialog_act_trace", {}) if isinstance(diagnostics.get("dialog_act_trace", {}), dict) else {}
        extra_patch = dialog_act_trace.get("extra_state_patch", {}) if isinstance(dialog_act_trace.get("extra_state_patch", {}), dict) else {}
        product_context = diagnostics.get("product_context_trace", {}) if isinstance(diagnostics.get("product_context_trace", {}), dict) else {}

        raw_mentions = self._string_list(slots.get("raw_brand_mentions", []))
        recognized = self._normalized_list(slots.get("brands", []))
        unknown_surfaces = self._string_list(slots.get("unknown_brand_mentions", []))
        product_hint = self._resolve_product_hint(extra_patch=extra_patch, product_context=product_context, default="tis")

        if raw_mentions:
            flow = self._apply_brand_list_received(
                flow=flow,
                raw_brand_mentions=raw_mentions,
                recognized_brands=recognized,
                unknown_brand_mentions=unknown_surfaces,
                product_hint=product_hint,
            )
        elif self._should_activate_pricing_flow(topic_result=topic_result, product_context=product_context):
            flow.active = True
            flow.product = product_hint or flow.product or "tis"
            if not flow.stage or flow.stage == "idle":
                flow.stage = "collect_brands"

        pricing_mode = str(extra_patch.get("pricing_mode", "")).strip() or str(product_context.get("pricing_mode", "")).strip()
        if pricing_mode == PricingFlowMode.SEPARATE.value:
            flow = self._apply_mode(flow, PricingFlowEvent.SEPARATE_REQUESTED)
        elif pricing_mode == PricingFlowMode.REMAINING_ONLY.value:
            flow = self._apply_mode(flow, PricingFlowEvent.REMAINING_REQUESTED)
        elif pricing_mode == PricingFlowMode.EXPLAIN_UNRESOLVED.value:
            flow = self._apply_mode(flow, PricingFlowEvent.CLARIFICATION_PUSHBACK)

        return PricingFlowPatch(flow=self._sync_derived_fields(flow))

    def apply_response_event(
        self,
        *,
        state_before_response: SessionState,
        response_action: str,
        answer_block: dict[str, object] | None = None,
    ) -> PricingFlowPatch:
        flow = self.from_session_state(state_before_response)
        block = answer_block if isinstance(answer_block, dict) else {}
        price_context = block.get("price_context", {}) if isinstance(block.get("price_context", {}), dict) else {}

        if response_action == "human_operator":
            flow = self._apply_mode(flow, PricingFlowEvent.HUMAN_HANDOFF)
            return PricingFlowPatch(flow=self._sync_derived_fields(flow))

        if response_action == "pricing_summary":
            flow.active = True
            flow.product = flow.product or "tis"
            if not flow.stage or flow.stage == "idle":
                flow.stage = "collect_brands"
            return PricingFlowPatch(flow=self._sync_derived_fields(flow))

        if response_action != "tis_tariffs":
            return PricingFlowPatch(flow=self._sync_derived_fields(flow))

        flow.active = True
        flow.product = flow.product or "tis"
        priced_brands = set(self._normalized_list(price_context.get("priced_brands", [])))
        missing_brands = set(self._normalized_list(price_context.get("missing_tis_price_brands", [])))
        recognized_brands = set(self._normalized_list(price_context.get("recognized_brands", [])))
        rendered_priced = set(self._normalized_list(price_context.get("rendered_priced_brands", [])))
        rendered_missing = set(self._normalized_list(price_context.get("rendered_missing_tis_price_brands", [])))

        for item in flow.brand_items:
            if not item.recognized:
                item.status = PricingBrandStatus.UNRESOLVED
                item.has_price = False
                continue
            canonical = item.canonical_brand
            if recognized_brands and canonical not in recognized_brands and canonical not in priced_brands and canonical not in missing_brands:
                continue
            if canonical in priced_brands:
                item.has_price = True
                item.status = PricingBrandStatus.PRICED
                if canonical in rendered_priced:
                    item.processed = True
            elif canonical in missing_brands:
                item.has_price = False
                item.status = PricingBrandStatus.MISSING_PRICE
                if canonical in rendered_missing:
                    item.processed = True

        return PricingFlowPatch(flow=self._sync_derived_fields(flow))

    def build_from_turn(
        self,
        *,
        session_state: SessionState,
        merged_slots: dict[str, object],
        pricing_mode: str = "",
        product_hint: str = "tis",
    ) -> PricingFlowState:
        flow = self.from_session_state(session_state)
        raw_mentions = self._string_list(merged_slots.get("raw_brand_mentions", []))
        recognized = self._normalized_list(merged_slots.get("brands", []))
        unknown_surfaces = self._string_list(merged_slots.get("unknown_brand_mentions", []))
        if raw_mentions:
            flow = self._apply_brand_list_received(
                flow=flow,
                raw_brand_mentions=raw_mentions,
                recognized_brands=recognized,
                unknown_brand_mentions=unknown_surfaces,
                product_hint=product_hint,
            )
        if pricing_mode:
            if pricing_mode == PricingFlowMode.SEPARATE.value:
                flow = self._apply_mode(flow, PricingFlowEvent.SEPARATE_REQUESTED)
            elif pricing_mode == PricingFlowMode.REMAINING_ONLY.value:
                flow = self._apply_mode(flow, PricingFlowEvent.REMAINING_REQUESTED)
            elif pricing_mode == PricingFlowMode.EXPLAIN_UNRESOLVED.value:
                flow = self._apply_mode(flow, PricingFlowEvent.CLARIFICATION_PUSHBACK)
            else:
                flow.mode = self._parse_mode(pricing_mode)
        return self._sync_derived_fields(flow)

    def _apply_brand_list_received(
        self,
        *,
        flow: PricingFlowState,
        raw_brand_mentions: list[str],
        recognized_brands: list[str],
        unknown_brand_mentions: list[str],
        product_hint: str,
    ) -> PricingFlowState:
        flow.active = True
        flow.product = str(product_hint or flow.product or "tis").strip().lower() or "tis"
        flow.stage = "collect_brands"
        flow.brand_mentions = self._build_brand_mentions(
            raw_brand_mentions=raw_brand_mentions,
            recognized_brands=recognized_brands,
            unknown_brand_mentions=unknown_brand_mentions,
        )
        flow.brand_items = self._build_brand_items(flow.brand_mentions)
        return flow

    def _apply_mode(self, flow: PricingFlowState, event_type: str) -> PricingFlowState:
        flow.active = True
        if not flow.product:
            flow.product = "tis"
        if event_type == PricingFlowEvent.SEPARATE_REQUESTED:
            flow.mode = PricingFlowMode.SEPARATE
            flow.stage = "pricing_separate"
        elif event_type == PricingFlowEvent.REMAINING_REQUESTED:
            flow.mode = PricingFlowMode.REMAINING_ONLY
            flow.stage = "unresolved_followup"
        elif event_type == PricingFlowEvent.CLARIFICATION_PUSHBACK:
            flow.mode = PricingFlowMode.EXPLAIN_UNRESOLVED
            flow.stage = "unresolved_followup"
        elif event_type == PricingFlowEvent.HUMAN_HANDOFF:
            flow.mode = PricingFlowMode.ALL
            flow.stage = "manager_handoff"
        return flow

    def _build_brand_mentions(
        self,
        *,
        raw_brand_mentions: list[str],
        recognized_brands: list[str],
        unknown_brand_mentions: list[str],
    ) -> list[BrandMention]:
        mentions: list[BrandMention] = []
        recognized_set = set(recognized_brands)
        for raw_surface in raw_brand_mentions:
            normalized_key = self._normalize_key(raw_surface)
            canonical = self._brand_resolver.canonical_for(normalized_key)
            display_name = self._display_surface(raw_surface)
            if canonical and canonical in recognized_set:
                display_name = self._brand_resolver.display_name_for(canonical) or display_name
                mentions.append(
                    BrandMention(
                        raw_text=str(raw_surface).strip(),
                        normalized_key=normalized_key,
                        recognized=True,
                        display_name=self._display_surface(display_name),
                        canonical_brand=canonical,
                    )
                )
            else:
                mentions.append(
                    BrandMention(
                        raw_text=str(raw_surface).strip(),
                        normalized_key=normalized_key,
                        recognized=False,
                        display_name=self._display_surface(raw_surface),
                        canonical_brand="",
                    )
                )
        if not mentions:
            for surface in unknown_brand_mentions:
                normalized_key = self._normalize_key(surface)
                mentions.append(
                    BrandMention(
                        raw_text=str(surface).strip(),
                        normalized_key=normalized_key,
                        recognized=False,
                        display_name=self._display_surface(surface),
                        canonical_brand="",
                    )
                )
        return mentions

    def _build_brand_items(self, brand_mentions: list[BrandMention]) -> list[PricingBrandItem]:
        items: list[PricingBrandItem] = []
        seen_recognized: set[str] = set()
        seen_unresolved: set[str] = set()
        for mention in brand_mentions:
            if mention.recognized:
                canonical = mention.canonical_brand
                if not canonical or canonical in seen_recognized:
                    continue
                seen_recognized.add(canonical)
                items.append(
                    PricingBrandItem(
                        canonical_brand=canonical,
                        display_name=self._brand_resolver.display_name_for(canonical),
                        raw_surface=mention.raw_text or mention.display_name,
                        recognized=True,
                        has_price=False,
                        processed=False,
                        status=PricingBrandStatus.MISSING_PRICE,
                    )
                )
                continue
            normalized = mention.normalized_key
            if not normalized or normalized in seen_unresolved:
                continue
            seen_unresolved.add(normalized)
            items.append(
                PricingBrandItem(
                    canonical_brand="",
                    display_name=self._display_surface(mention.raw_text or mention.display_name),
                    raw_surface=mention.raw_text or mention.display_name,
                    recognized=False,
                    has_price=False,
                    processed=False,
                    status=PricingBrandStatus.UNRESOLVED,
                )
            )
        return items

    def _build_brand_items_from_compatibility(
        self,
        *,
        requested_brand_keys: list[str],
        recognized_brands: list[str],
        unknown_brand_mentions: list[str],
        priced_brands: list[str],
        missing_price_brands: list[str],
    ) -> list[PricingBrandItem]:
        items: list[PricingBrandItem] = []
        for canonical in recognized_brands:
            status = PricingBrandStatus.PRICED if canonical in priced_brands else PricingBrandStatus.MISSING_PRICE
            items.append(
                PricingBrandItem(
                    canonical_brand=canonical,
                    display_name=self._brand_resolver.display_name_for(canonical),
                    raw_surface=self._brand_resolver.display_name_for(canonical),
                    recognized=True,
                    has_price=canonical in priced_brands,
                    processed=canonical in priced_brands,
                    status=status,
                )
            )
        for surface in unknown_brand_mentions:
            items.append(
                PricingBrandItem(
                    canonical_brand="",
                    display_name=self._display_surface(surface),
                    raw_surface=str(surface).strip(),
                    recognized=False,
                    has_price=False,
                    processed=False,
                    status=PricingBrandStatus.UNRESOLVED,
                )
            )
        if requested_brand_keys and not items:
            for key in requested_brand_keys:
                canonical = self._brand_resolver.canonical_for(key)
                if canonical:
                    status = PricingBrandStatus.PRICED if canonical in priced_brands else PricingBrandStatus.MISSING_PRICE
                    items.append(
                        PricingBrandItem(
                            canonical_brand=canonical,
                            display_name=self._brand_resolver.display_name_for(canonical),
                            raw_surface=key,
                            recognized=True,
                            has_price=canonical in priced_brands,
                            processed=canonical in priced_brands,
                            status=status,
                        )
                    )
                else:
                    items.append(
                        PricingBrandItem(
                            canonical_brand="",
                            display_name=self._display_surface(key),
                            raw_surface=key,
                            recognized=False,
                            has_price=False,
                            processed=False,
                            status=PricingBrandStatus.UNRESOLVED,
                        )
                    )
        return items

    @classmethod
    def _sync_derived_fields(cls, flow: PricingFlowState) -> PricingFlowState:
        raw_mentions: list[str] = []
        requested_keys: list[str] = []
        recognized: list[str] = []
        unknown: list[str] = []
        priced: list[str] = []
        missing: list[str] = []
        pending: list[str] = []
        processed: list[str] = []
        remaining: list[str] = []
        brand_mentions: list[BrandMention] = []

        for item in flow.brand_items:
            raw_surface = str(item.raw_surface).strip()
            display_name = str(item.display_name).strip() or cls._display_surface(raw_surface)
            normalized_key = cls._normalize_key(raw_surface or display_name)
            if raw_surface and raw_surface not in raw_mentions:
                raw_mentions.append(raw_surface)
            if normalized_key and normalized_key not in requested_keys:
                requested_keys.append(normalized_key)
            if item.recognized:
                canonical = str(item.canonical_brand).strip().lower()
                if canonical and canonical not in recognized:
                    recognized.append(canonical)
                brand_mentions.append(
                    BrandMention(
                        raw_text=raw_surface or display_name,
                        normalized_key=normalized_key or canonical,
                        recognized=True,
                        display_name=display_name,
                        canonical_brand=canonical,
                    )
                )
                if item.status == PricingBrandStatus.PRICED:
                    if canonical and canonical not in priced:
                        priced.append(canonical)
                elif item.status == PricingBrandStatus.MISSING_PRICE:
                    if canonical and canonical not in missing:
                        missing.append(canonical)
                    if display_name and display_name not in pending:
                        pending.append(display_name)
                if item.processed and display_name not in processed:
                    processed.append(display_name)
                if not item.processed and display_name not in remaining:
                    remaining.append(display_name)
                continue

            display_surface = display_name or cls._display_surface(raw_surface)
            if display_surface and display_surface not in unknown:
                unknown.append(display_surface)
            brand_mentions.append(
                BrandMention(
                    raw_text=raw_surface or display_surface,
                    normalized_key=normalized_key,
                    recognized=False,
                    display_name=display_surface,
                    canonical_brand="",
                )
            )
            if item.processed:
                if display_surface and display_surface not in processed:
                    processed.append(display_surface)
            else:
                if display_surface and display_surface not in pending:
                    pending.append(display_surface)
                if display_surface and display_surface not in remaining:
                    remaining.append(display_surface)

        flow.brand_mentions = brand_mentions
        flow.raw_brand_mentions = raw_mentions
        flow.requested_brand_keys = requested_keys
        flow.recognized_brands = recognized
        flow.unknown_brand_mentions = unknown
        flow.priced_brands = priced
        flow.missing_price_brands = missing
        flow.pending_brand_mentions = pending
        flow.processed_brand_mentions = processed
        flow.remaining_brand_mentions = remaining
        if flow.active and not flow.stage:
            flow.stage = "pricing_all"
        elif flow.active and flow.stage == "collect_brands":
            flow.stage = "pricing_all"
        return flow

    def _parse_brand_mentions(self, value: object) -> list[BrandMention]:
        if not isinstance(value, list):
            return []
        mentions: list[BrandMention] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            mentions.append(
                BrandMention(
                    raw_text=str(item.get("raw_text", "")).strip(),
                    normalized_key=str(item.get("normalized_key", "")).strip(),
                    recognized=bool(item.get("recognized", False)),
                    display_name=str(item.get("display_name", "")).strip(),
                    canonical_brand=str(item.get("canonical_brand", "")).strip(),
                )
            )
        return mentions

    def _parse_brand_items(self, value: object, brand_mentions: list[BrandMention]) -> list[PricingBrandItem]:
        if isinstance(value, list) and value:
            items: list[PricingBrandItem] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                items.append(
                    PricingBrandItem(
                        canonical_brand=str(item.get("canonical_brand", "")).strip().lower(),
                        display_name=self._display_surface(item.get("display_name", "")),
                        raw_surface=str(item.get("raw_surface", "")).strip(),
                        recognized=bool(item.get("recognized", False)),
                        has_price=bool(item.get("has_price", False)),
                        processed=bool(item.get("processed", False)),
                        status=self._parse_status(item.get("status", PricingBrandStatus.UNRESOLVED.value)),
                    )
                )
            return items
        if brand_mentions:
            return self._build_brand_items(brand_mentions)
        return []

    @staticmethod
    def _should_activate_pricing_flow(
        *,
        topic_result: TopicClassificationResult,
        product_context: dict[str, object],
    ) -> bool:
        inferred = str(product_context.get("inferred_product_context", "")).strip().lower()
        asks_price = bool(product_context.get("asks_price", False))
        topic_ids = [str(item).strip() for item in topic_result.topic_ids if str(item).strip()]
        pricing_topics = {"epc_tariffs", "tis_tariffs", "purchase_ready"}
        return asks_price or inferred == "tis" or any(topic in pricing_topics for topic in topic_ids)

    @staticmethod
    def _resolve_product_hint(*, extra_patch: dict[str, object], product_context: dict[str, object], default: str) -> str:
        for key in ("pricing_requested_product", "active_pricing_flow"):
            value = str(extra_patch.get(key, "")).strip().lower()
            if value and value != "none":
                return value
        value = str(product_context.get("inferred_product_context", "")).strip().lower()
        if value:
            return value
        return default

    @staticmethod
    def _parse_mode(value: object) -> PricingFlowMode:
        raw = str(value or "").strip()
        for mode in PricingFlowMode:
            if raw == mode.value:
                return mode
        return PricingFlowMode.ALL

    @staticmethod
    def _parse_status(value: object) -> PricingBrandStatus:
        raw = str(value or "").strip()
        for status in PricingBrandStatus:
            if raw == status.value:
                return status
        return PricingBrandStatus.UNRESOLVED

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _normalized_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            normalized = str(item).strip().lower()
            if normalized and normalized not in items:
                items.append(normalized)
        return items

    @staticmethod
    def _normalize_key(value: object) -> str:
        return str(value).strip().lower().replace("ё", "е")

    @staticmethod
    def _display_surface(value: object) -> str:
        surface = str(value).strip()
        if not surface:
            return ""
        if surface.islower() and surface.isascii():
            return surface.capitalize()
        return surface
