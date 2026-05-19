from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.domain.brand_display_resolver import BrandDisplayResolver
from src.domain.brands import BrandAliasResolver


@dataclass(slots=True)
class ProductContext:
    raw_query: str = ""
    requested_products: list[str] = field(default_factory=list)
    raw_brand_mentions: list[str] = field(default_factory=list)
    mentioned_brands: list[str] = field(default_factory=list)
    unknown_brand_mentions: list[str] = field(default_factory=list)
    brand_display: list[str] = field(default_factory=list)
    asks_price: bool = False
    asks_availability: bool = False
    asks_comparison: bool = False
    partial_access_requested: bool = False
    partial_package_requested: bool = False
    partial_epc_requested: bool = False
    inferred_product_context: str = ""
    unsupported_brand_group: str = ""
    unknown_brand_query: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_query": self.raw_query,
            "requested_products": list(self.requested_products),
            "raw_brand_mentions": list(self.raw_brand_mentions),
            "mentioned_brands": list(self.mentioned_brands),
            "unknown_brand_mentions": list(self.unknown_brand_mentions),
            "brand_display": list(self.brand_display),
            "asks_price": self.asks_price,
            "asks_availability": self.asks_availability,
            "asks_comparison": self.asks_comparison,
            "partial_access_requested": self.partial_access_requested,
            "partial_package_requested": self.partial_package_requested,
            "partial_epc_requested": self.partial_epc_requested,
            "inferred_product_context": self.inferred_product_context,
            "unsupported_brand_group": self.unsupported_brand_group,
            "unknown_brand_query": self.unknown_brand_query,
        }


class ProductResolver:
    _TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9-]+")
    _PRICE_PHRASES = (
        "что по деньгам",
        "по деньгам",
        "сколько стоит",
        "какая стоимость",
        "какие тарифы",
        "цена подписки",
    )
    _AVAILABILITY_PHRASES = (
        "есть ли",
        "поддерживается",
        "какой каталог",
        "наличие",
        "доступен ли",
        "есть каталог",
    )
    _COMPARISON_TOKENS = {"разница", "отлич", "входит", "отдельно", "состав"}
    _FOLLOWUP_PRICE_TOKENS = {"только", "именно", "по нему", "по ней", "по ним", "подскажи", "подсказывай"}
    _NON_BRAND_FOLLOWUPS = {
        "что уточнить",
        "это ты мне уточни",
        "что именно уточнить",
        "ты должен уточнить",
        "ты сам уточни",
        "сколько можно",
        "а остальные",
        "остальные",
        "а по другим",
        "а еще",
    }
    _PARTIAL_ACCESS_PHRASES = (
        "не полный пакет",
        "часть доступа",
        "не весь пакет",
        "не полный доступ",
        "часть каталога",
        "отдельно по брендам",
        "полный пакет",
    )
    _PARTIAL_PACKAGE_PHRASES = (
        "только один бренд",
        "только одного бренда",
        "каталоги только одного бренда",
        "предоставить каталоги только одного бренда",
        "есть ли возможность предоставить каталоги только одного бренда",
        "возможность предоставить каталоги только одного бренда",
        "купить только один бренд",
        "одного бренда",
        "одна марка",
        "только одна марка",
        "один бренд в epc",
        "один бренд в епс",
        "одну марку в epc",
        "одну марку в епс",
    )
    _UNSUPPORTED_GROUPS = {
        "vag": "VAG",
        "ваг": "VAG",
    }
    _UNKNOWN_BRAND_PATTERNS = (
        re.compile(r"\b(?:на|для|по|марка|бренд)\s+([a-zа-яё][a-zа-яё0-9-]{1,30})", re.IGNORECASE),
        re.compile(r"\bтолько\s+([a-zа-яё][a-zа-яё0-9-]{1,30})\b", re.IGNORECASE),
        re.compile(r"\bкаталог\s+([a-zа-яё][a-zа-яё0-9-]{1,30})\b", re.IGNORECASE),
    )
    _BRAND_STOPWORDS = {
        "каталог",
        "каталоги",
        "бренд",
        "бренда",
        "марка",
        "марки",
        "доступ",
        "подписка",
        "пакет",
        "пользователя",
        "пользователь",
        "сотрудника",
        "сотрудников",
        "мне",
        "вам",
        "одного",
        "один",
        "одна",
        "нужен",
        "нужна",
        "можно",
    }

    def __init__(self, brands_file_path: Path) -> None:
        self._brand_resolver = BrandAliasResolver(brands_file_path)
        self._brand_display = BrandDisplayResolver()

    def resolve(
        self,
        *,
        user_query: str,
        history_text: str = "",
        state_snapshot: dict[str, object] | None = None,
        slot_trace: dict[str, object] | None = None,
    ) -> ProductContext:
        normalized_query = self._normalize(user_query)
        normalized_history = self._normalize(history_text)
        query_tokens = set(self._tokenize(normalized_query))
        slots = slot_trace.get("slots", {}) if isinstance(slot_trace, dict) and isinstance(slot_trace.get("slots", {}), dict) else {}

        requested_products: list[str] = []
        if "epc" in query_tokens:
            requested_products.append("epc")
        if "tis" in query_tokens:
            requested_products.append("tis")

        raw_brand_mentions = slots.get("raw_brand_mentions", [])
        if not isinstance(raw_brand_mentions, list):
            raw_brand_mentions = []
        mentioned_brands = slots.get("brands", [])
        if isinstance(mentioned_brands, list):
            mentioned_brands = [str(item).strip().lower() for item in mentioned_brands if str(item).strip()]
        else:
            mentioned_brands = list(self._brand_resolver.extract(user_query))
        unknown_brand_mentions = slots.get("unknown_brand_mentions", [])
        if isinstance(unknown_brand_mentions, list):
            unknown_brand_mentions = [str(item).strip() for item in unknown_brand_mentions if str(item).strip()]
        else:
            unknown_brand_mentions = []
        unsupported_brand_group = self._detect_unsupported_brand_group(query_tokens)
        if not mentioned_brands:
            last_brand = str((state_snapshot or {}).get("last_mentioned_brand", "")).strip().lower()
            if last_brand and self._is_followup_brand_reply(normalized_query):
                mentioned_brands.append(last_brand)
        slot_brand_display = slots.get("brand_display_list", [])
        if isinstance(slot_brand_display, list) and slot_brand_display:
            brand_display = [str(item).strip() for item in slot_brand_display if str(item).strip()]
        else:
            brand_display = [self._brand_display.display(item) for item in mentioned_brands if str(item).strip()]

        asks_price = self._asks_price(
            normalized_query=normalized_query,
            normalized_history=normalized_history,
            mentioned_brands=mentioned_brands,
        )
        asks_comparison = bool({"epc", "tis"} & query_tokens) and any(
            token in normalized_query for token in self._COMPARISON_TOKENS
        )
        partial_package_requested = any(phrase in normalized_query for phrase in self._PARTIAL_PACKAGE_PHRASES)
        partial_access_requested = partial_package_requested or any(
            phrase in normalized_query for phrase in self._PARTIAL_ACCESS_PHRASES
        )
        asks_availability = self._asks_availability(
            normalized_query=normalized_query,
            query_tokens=query_tokens,
            partial_package_requested=partial_package_requested,
        )
        unknown_brand_query = ""
        if not mentioned_brands and not unsupported_brand_group and (asks_availability or asks_price):
            unknown_brand_query = self._extract_unknown_brand_candidate(normalized_query)

        inferred_product_context = ""
        if requested_products:
            inferred_product_context = requested_products[0]
        elif mentioned_brands:
            inferred_product_context = "tis"

        return ProductContext(
            raw_query=user_query,
            requested_products=requested_products,
            raw_brand_mentions=[str(item).strip() for item in raw_brand_mentions if str(item).strip()],
            mentioned_brands=mentioned_brands,
            unknown_brand_mentions=unknown_brand_mentions,
            brand_display=brand_display,
            asks_price=asks_price,
            asks_availability=asks_availability,
            asks_comparison=asks_comparison,
            partial_access_requested=partial_access_requested,
            partial_package_requested=partial_package_requested,
            partial_epc_requested=partial_package_requested,
            inferred_product_context=inferred_product_context,
            unsupported_brand_group=unsupported_brand_group,
            unknown_brand_query=unknown_brand_query,
        )

    def _asks_price(
        self,
        *,
        normalized_query: str,
        normalized_history: str,
        mentioned_brands: list[str],
    ) -> bool:
        if any(phrase in normalized_query for phrase in self._PRICE_PHRASES):
            return True
        query_tokens = set(self._tokenize(normalized_query))
        if {"цена", "стоимость", "тариф", "тарифы", "подписка", "сколько"} & query_tokens:
            return True
        if mentioned_brands and self._is_followup_brand_reply(normalized_query):
            return any(token in normalized_history for token in ("стоим", "цена", "тариф", "tis"))
        return False

    def _asks_availability(
        self,
        *,
        normalized_query: str,
        query_tokens: set[str],
        partial_package_requested: bool,
    ) -> bool:
        if partial_package_requested:
            return False
        if "каталог" in query_tokens and "есть ли" in normalized_query:
            return True
        return any(phrase in normalized_query for phrase in self._AVAILABILITY_PHRASES)

    def _detect_unsupported_brand_group(self, query_tokens: set[str]) -> str:
        for token in query_tokens:
            value = self._UNSUPPORTED_GROUPS.get(token)
            if value:
                return value
        return ""

    def _extract_unknown_brand_candidate(self, normalized_query: str) -> str:
        for pattern in self._UNKNOWN_BRAND_PATTERNS:
            match = pattern.search(normalized_query)
            if not match:
                continue
            candidate = str(match.group(1)).strip().lower()
            if candidate and candidate not in self._BRAND_STOPWORDS:
                return candidate
        return ""

    def _is_followup_brand_reply(self, normalized_query: str) -> bool:
        if not normalized_query:
            return False
        if normalized_query in self._NON_BRAND_FOLLOWUPS:
            return False
        if any(token in normalized_query for token in self._FOLLOWUP_PRICE_TOKENS):
            return True
        return len(self._tokenize(normalized_query)) <= 2

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [match.group(0) for match in cls._TOKEN_RE.finditer(text)]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()
