from __future__ import annotations

from dataclasses import dataclass, field

from src.core.models import MandatoryMeaningBlock


@dataclass(slots=True)
class MeaningValidationResult:
    is_valid: bool
    block_results: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "block_results": dict(self.block_results),
        }


class MeaningValidator:
    _TAG_RULES = {
        "tis_price_missing": (
            ["tis", "цена tis", "для tis"],
            ["не указана", "не найдена", "нет в прайсе", "в текущем прайсе не указана", "отсутствует"],
        ),
        "epc_fallback_available": (
            ["epc full", "epc"],
            ["тариф", "1 месяц", "можно рассмотреть", "доступен", "доступны"],
        ),
        "human_operator_handoff": (
            ["менеджер", "специалист", "свяжется", "передам", "подключу человека"],
        ),
    }

    def validate(self, candidate_answer: str, mandatory_blocks: list[MandatoryMeaningBlock]) -> MeaningValidationResult:
        normalized = self._normalize(candidate_answer)
        if not mandatory_blocks:
            return MeaningValidationResult(is_valid=bool(normalized), block_results={})
        block_results: dict[str, bool] = {}
        for block in mandatory_blocks:
            block_results[block.key] = self._validate_block(normalized, block)
        return MeaningValidationResult(is_valid=all(block_results.values()), block_results=block_results)

    def _validate_block(self, normalized_answer: str, block: MandatoryMeaningBlock) -> bool:
        if not normalized_answer:
            return False
        semantic_pass = True
        if block.semantic_tags:
            semantic_pass = all(self._match_tag(normalized_answer, tag) for tag in block.semantic_tags)
        phrase_pass = True
        if block.required_phrases:
            phrase_pass = all(self._normalize(phrase) in normalized_answer for phrase in block.required_phrases)
        return semantic_pass or phrase_pass

    def _match_tag(self, normalized_answer: str, tag: str) -> bool:
        groups = self._TAG_RULES.get(str(tag).strip(), ())
        if not groups:
            return False
        return all(any(self._normalize(option) in normalized_answer for option in group) for group in groups)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").lower().replace("ё", "е").split())
