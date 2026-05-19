from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.policy.types import ResponseAction


@dataclass(slots=True)
class ActionValidationResult:
    is_valid: bool
    blocker_hits: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "blocker_hits": list(self.blocker_hits),
            "missing_required": list(self.missing_required),
        }


class ActionCompatibilityValidator:
    _BLOCKERS = {
        "existing_contract_check": ("инн", "количество доступ", "qr", "телефон, имя", "желаемый период"),
        "human_operator": ("уточните стоимость", "уточните цену", "уточните тариф", "напишите бренды", "пожалуйста, уточните стоимость"),
        "company_services": ("демо-доступ", "подтвердите юридический статус", "оформление демо"),
        "tis_tariffs": ("уточните стоимость",),
        "ask_legal_status": ("6500", "18000", "34800", "62400"),
    }
    _REQUIRED = {
        "human_operator": ("менеджер", "специалист", "свяжется", "передам", "подключу человека"),
        "brand_group_clarification": ("vag", "уточните", "конкретную марку"),
        "partial_catalog_restriction": ("epc full", "полным пакетом", "tis"),
        "ask_legal_status": ("юрлиц", "ип"),
    }

    def validate(self, selected_action: ResponseAction, answer_text: str) -> ActionValidationResult:
        answer = " ".join(str(answer_text or "").lower().replace("ё", "е").split())
        if not answer:
            return ActionValidationResult(is_valid=False)
        blocker_hits = [token for token in self._BLOCKERS.get(selected_action.name, ()) if token in answer]
        required_tokens = self._REQUIRED.get(selected_action.name, ())
        missing_required = []
        if required_tokens and not any(token in answer for token in required_tokens):
            missing_required = list(required_tokens)
        return ActionValidationResult(
            is_valid=not blocker_hits and not missing_required,
            blocker_hits=blocker_hits,
            missing_required=missing_required,
        )
