from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.core.models import SessionState
from src.domain.brand_display_resolver import BrandDisplayResolver
from src.domain.brands import BrandAliasResolver


@dataclass(slots=True)
class SlotExtractionResult:
    slots: dict[str, object] = field(default_factory=dict)
    trace: dict[str, object] = field(default_factory=dict)


class SlotExtractionService:
    _PHONE_PATTERNS = (
        re.compile(r"(?<!\d)(?:\+7|8)\s*\(?\d{3}\)?[\s-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"),
    )
    _INN_PATTERNS = (
        re.compile(r"(?<!\d)\d{8}(?!\d)"),
        re.compile(r"(?<!\d)\d{10}(?!\d)"),
        re.compile(r"(?<!\d)\d{12}(?!\d)"),
    )
    _CONTRACT_PATTERNS = (
        re.compile(r"(?:договор[а-я\s№#:]*)(\d[\d/-]*)", re.IGNORECASE),
        re.compile(r"(?:номер[а-я\s]*договора[а-я\s:]*)(\d[\d/-]*)", re.IGNORECASE),
    )
    _PERIOD_PATTERNS = (
        ("12_months", re.compile(r"\b(?:1\s*год|12\s*месяц(?:ев|а)?)\b", re.IGNORECASE)),
        ("6_months", re.compile(r"\b6\s*месяц(?:ев|а)?\b", re.IGNORECASE)),
        ("3_months", re.compile(r"\b3\s*месяц(?:ев|а)?\b", re.IGNORECASE)),
        ("1_month", re.compile(r"\b1\s*месяц\b", re.IGNORECASE)),
    )
    _USER_COUNT_PATTERNS = (
        (1, re.compile(r"\b(?:1|один|одного)\s*(?:пользоват|сотрудник|доступ)\w*", re.IGNORECASE)),
        (2, re.compile(r"\b(?:2|двух|два|двоим)\s*(?:пользоват|сотрудник|доступ)\w*", re.IGNORECASE)),
        (3, re.compile(r"\b(?:3|трех|трёх|три)\s*(?:пользоват|сотрудник|доступ)\w*", re.IGNORECASE)),
    )
    _SELF_EMPLOYED_PATTERNS = (
        re.compile(r"\bсамозанят\w*\b", re.IGNORECASE),
    )
    _LEGAL_PATTERNS = {
        "ip": re.compile(r"\b(?:ип|индивидуальн\w+\s+предпринимател\w*)\b", re.IGNORECASE),
        "legal_entity": re.compile(r"\b(?:юр\.?\s*лиц|юридическ\w+\s+лиц\w*|ооо|зао|ао)\b", re.IGNORECASE),
    }
    _PAYMENT_PATTERNS = {
        "invoice": re.compile(r"\b(?:по\s+счету|по\s+сч[её]ту|счет|сч[её]т|выставить\s+счет)\b", re.IGNORECASE),
        "qr": re.compile(r"\b(?:qr|кьюар|по\s+qr)\b", re.IGNORECASE),
        "card": re.compile(r"\b(?:картой|по\s+карте|карта)\b", re.IGNORECASE),
    }

    def __init__(self, brands_file_path: Path) -> None:
        self._brand_resolver = BrandAliasResolver(brands_file_path)
        self._brand_display = BrandDisplayResolver()

    def extract(self, user_query: str, session_state: SessionState | None = None) -> SlotExtractionResult:
        query = str(user_query or "")
        slots: dict[str, object] = {}
        matched_patterns: dict[str, str] = {}

        phone = self._extract_phone(query)
        if phone:
            slots["phone"] = phone
            matched_patterns["phone"] = "phone_regex"

        inn = self._extract_inn(query)
        if inn:
            slots["inn"] = inn
            matched_patterns["inn"] = "inn_regex"

        period = self._extract_period(query)
        if period:
            slots["period"] = period
            matched_patterns["period"] = "period_regex"

        user_count = self._extract_user_count(query)
        if user_count is not None:
            slots["user_count"] = user_count
            matched_patterns["user_count"] = "user_count_regex"

        payment_method = self._extract_payment_method(query)
        if payment_method:
            slots["payment_method"] = payment_method
            matched_patterns["payment_method"] = "payment_regex"

        products = self._extract_products(query)
        if products:
            slots["products"] = list(products)
            matched_patterns["products"] = "product_keyword_scan"

        contract_number = self._extract_contract_number(query)
        if contract_number:
            slots["contract_number"] = contract_number
            matched_patterns["contract_number"] = "contract_regex"

        legal_status = self._extract_legal_status(query)
        if legal_status:
            slots["legal_status"] = legal_status
            matched_patterns["legal_status"] = "legal_status_regex"

        if self._contains_self_employed(query):
            slots["self_employed_signal"] = True
            matched_patterns["self_employed_signal"] = "self_employed_regex"

        brand_result = self._brand_resolver.extract_detailed(query)
        if brand_result.raw_mentions:
            slots["raw_brand_mentions"] = list(brand_result.raw_mentions)
            matched_patterns["raw_brand_mentions"] = "brand_alias_resolver"
        if brand_result.brand_mentions:
            slots["brand_mentions"] = list(brand_result.brand_mentions)
            matched_patterns["brand_mentions"] = "brand_alias_resolver"
        if brand_result.requested_brand_keys:
            slots["requested_brand_keys"] = list(brand_result.requested_brand_keys)
            matched_patterns["requested_brand_keys"] = "brand_alias_resolver"
        if brand_result.unknown_brand_mentions:
            slots["unknown_brand_mentions"] = list(brand_result.unknown_brand_mentions)
            matched_patterns["unknown_brand_mentions"] = "brand_alias_resolver"
        brands = list(brand_result.recognized_brands)
        if brands:
            display_values = [self._brand_display.display(str(item)) for item in brands]
            slots["brands"] = list(brands)
            slots["brand"] = str(brands[0]).strip().lower()
            slots["brand_display"] = display_values[0]
            slots["brand_display_list"] = display_values
            matched_patterns["brand"] = "brand_alias_resolver"
        else:
            inherited = str(session_state.last_mentioned_brand).strip().lower() if session_state else ""
            if inherited:
                slots["last_brand_source"] = "session"

        missing_slots = [
            name for name in ("phone", "inn", "period", "brand", "user_count", "payment_method")
            if name not in slots
        ]

        trace = {
            "input": query,
            "slots": dict(slots),
            "brand_mentions": list(brand_result.raw_mentions),
            "recognized_brands": list(brands),
            "unknown_brands": list(brand_result.unknown_brand_mentions),
            "matched_patterns": matched_patterns,
            "missing_slots": missing_slots,
        }
        return SlotExtractionResult(slots=slots, trace=trace)

    @classmethod
    def _extract_phone(cls, text: str) -> str:
        for pattern in cls._PHONE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            digits = re.sub(r"\D", "", match.group(0))
            if digits.startswith("8") and len(digits) == 11:
                digits = "7" + digits[1:]
            if len(digits) == 11 and digits.startswith("7"):
                return f"+{digits}"
        return ""

    @classmethod
    def _extract_inn(cls, text: str) -> str:
        for pattern in cls._INN_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return ""

    @classmethod
    def _extract_period(cls, text: str) -> str:
        for value, pattern in cls._PERIOD_PATTERNS:
            if pattern.search(text):
                return value
        return ""

    @classmethod
    def _extract_user_count(cls, text: str) -> int | None:
        for value, pattern in cls._USER_COUNT_PATTERNS:
            if pattern.search(text):
                return value
        if re.search(r"\bнескольк\w*\s+(?:люд|сотрудник|пользоват)\w*", text, re.IGNORECASE):
            return 2
        return None

    @classmethod
    def _extract_payment_method(cls, text: str) -> str:
        for method, pattern in cls._PAYMENT_PATTERNS.items():
            if pattern.search(text):
                return method
        return ""

    @staticmethod
    def _extract_products(text: str) -> list[str]:
        normalized = str(text or "").lower().replace("ё", "е")
        products: list[str] = []
        if "epc" in normalized:
            products.append("epc")
        if "tis" in normalized:
            products.append("tis")
        return products

    @classmethod
    def _extract_contract_number(cls, text: str) -> str:
        for pattern in cls._CONTRACT_PATTERNS:
            match = pattern.search(text)
            if match:
                return str(match.group(1)).strip()
        return ""

    @classmethod
    def _extract_legal_status(cls, text: str) -> str:
        if cls._LEGAL_PATTERNS["ip"].search(text):
            return "ip"
        if cls._LEGAL_PATTERNS["legal_entity"].search(text):
            return "legal_entity"
        return ""

    @classmethod
    def _contains_self_employed(cls, text: str) -> bool:
        return any(pattern.search(text) for pattern in cls._SELF_EMPLOYED_PATTERNS)
