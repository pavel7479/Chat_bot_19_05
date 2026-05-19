from __future__ import annotations

from dataclasses import dataclass, field

from src.core.models import FactRecord
from src.domain.brand_display_resolver import BrandDisplayResolver


@dataclass(slots=True)
class FactRenderResult:
    text: str
    status: str
    missing_slots: list[str]
    trace: dict[str, object]


@dataclass(slots=True)
class FactTemplateRenderer:
    """Safely render structured fact templates from deterministic slots."""

    _brand_display: BrandDisplayResolver = field(default_factory=BrandDisplayResolver)

    def render(self, fact: FactRecord, slots: dict[str, object] | None = None) -> str:
        return self.render_with_trace(fact, slots).text

    def render_with_trace(self, fact: FactRecord, slots: dict[str, object] | None = None) -> FactRenderResult:
        slot_map = self._normalize_slots(slots or {})
        missing_slots = [
            name for name in fact.required_slots
            if name not in slot_map or not str(slot_map.get(name, "")).strip()
        ]
        if fact.template:
            if missing_slots:
                return FactRenderResult(
                    text="",
                    status="missing_slots",
                    missing_slots=missing_slots,
                    trace={
                        "fact_id": fact.fact_id,
                        "render_mode": "template",
                        "missing_slots": list(missing_slots),
                        "slot_keys": sorted(slot_map.keys()),
                    },
                )
            try:
                rendered = str(fact.template).format(**slot_map).strip()
            except Exception as exc:
                return FactRenderResult(
                    text="",
                    status="template_error",
                    missing_slots=[],
                    trace={
                        "fact_id": fact.fact_id,
                        "render_mode": "template",
                        "error": type(exc).__name__,
                        "slot_keys": sorted(slot_map.keys()),
                    },
                )
            return FactRenderResult(
                text=rendered,
                status="rendered",
                missing_slots=[],
                trace={
                    "fact_id": fact.fact_id,
                    "render_mode": "template",
                    "slot_keys": sorted(slot_map.keys()),
                },
            )
        return FactRenderResult(
            text=str(fact.text or "").strip(),
            status="rendered",
            missing_slots=[],
            trace={
                "fact_id": fact.fact_id,
                "render_mode": "text",
                "slot_keys": sorted(slot_map.keys()),
            },
        )

    def _normalize_slots(self, slots: dict[str, object]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        raw_brand = ""
        for key, value in slots.items():
            name = str(key).strip()
            if not name:
                continue
            text = str(value or "").strip()
            if not text:
                continue
            if name == "brand":
                raw_brand = text.lower()
                normalized[name] = raw_brand
            elif name == "brand_display":
                normalized[name] = self._brand_display.display(text)
            else:
                normalized[name] = text
        if raw_brand and "brand_display" not in normalized:
            normalized["brand_display"] = self._brand_display.display(raw_brand)
        return normalized
