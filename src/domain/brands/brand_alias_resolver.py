from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class BrandExtractionResult:
    raw_mentions: list[str] = field(default_factory=list)
    requested_brand_keys: list[str] = field(default_factory=list)
    recognized_brands: list[str] = field(default_factory=list)
    unknown_brand_mentions: list[str] = field(default_factory=list)
    brand_mentions: list[dict[str, object]] = field(default_factory=list)


class BrandAliasResolver:
    _BRAND_LIST_DELIMITER_RE = re.compile(r"[,;\n]")
    _NON_BRAND_SIGNAL_RE = re.compile(
        r"\b(?:инн|тел\.?|телефон|оплата|счет|сч[её]ту|пользоват|год|месяц|доступ|qr)\b",
        re.IGNORECASE,
    )

    def __init__(self, brands_file_path: Path) -> None:
        self._aliases_to_canonical, self._display_by_canonical = self._load(brands_file_path)
        self._aliases = sorted(self._aliases_to_canonical.keys(), key=len, reverse=True)

    @staticmethod
    def _load(brands_file_path: Path) -> tuple[dict[str, str], dict[str, str]]:
        raw = yaml.safe_load(brands_file_path.read_text(encoding="utf-8")) or {}
        aliases_to_canonical: dict[str, str] = {}
        display_by_canonical: dict[str, str] = {}
        for brand in raw.get("brands", []):
            canonical = str(brand.get("canonical", "")).strip().lower()
            if not canonical:
                continue
            aliases_to_canonical[canonical] = canonical
            display_by_canonical[canonical] = str(brand.get("display", "")).strip() or canonical.upper()
            for alias in brand.get("aliases", []):
                normalized = str(alias).strip().lower()
                if normalized:
                    aliases_to_canonical[normalized] = canonical
        return aliases_to_canonical, display_by_canonical

    def extract(self, text: str) -> list[str]:
        return self.extract_detailed(text).recognized_brands

    def extract_detailed(self, text: str) -> BrandExtractionResult:
        normalized = self._normalize_text(text)
        original_text = str(text or "")
        matched_mentions = self._extract_matched_mentions(original_text)
        if matched_mentions:
            return self._result_from_matched_mentions(matched_mentions)

        if self._looks_like_real_brand_list_input(original_text):
            raw_mentions = self._extract_raw_mentions(original_text)
            return self._result_from_raw_mentions(raw_mentions)

        return BrandExtractionResult()

    def _result_from_matched_mentions(self, matched_mentions: list[dict[str, str]]) -> BrandExtractionResult:
        recognized: list[str] = []
        brand_mentions: list[dict[str, object]] = []
        raw_mentions: list[str] = []
        requested_keys: list[str] = []
        for item in matched_mentions:
            raw_text = str(item.get("raw_text", "")).strip()
            normalized_key = str(item.get("normalized_key", "")).strip()
            canonical = str(item.get("canonical_brand", "")).strip().lower()
            if not raw_text or not canonical:
                continue
            raw_mentions.append(raw_text)
            requested_keys.append(normalized_key)
            if canonical not in recognized:
                recognized.append(canonical)
            brand_mentions.append(
                {
                    "raw_text": raw_text,
                    "normalized_key": normalized_key,
                    "recognized": True,
                    "display_name": str(item.get("display_name", "")).strip() or self.display_name_for(canonical),
                    "canonical_brand": canonical,
                }
            )
        return BrandExtractionResult(
            raw_mentions=raw_mentions,
            requested_brand_keys=requested_keys,
            recognized_brands=recognized,
            unknown_brand_mentions=[],
            brand_mentions=brand_mentions,
        )

    def canonical_for(self, alias_or_key: str) -> str:
        return self._aliases_to_canonical.get(self._normalize_text(alias_or_key), "")

    def display_name_for(self, canonical: str) -> str:
        key = self._normalize_text(canonical)
        return self._display_by_canonical.get(key, self._display_surface(canonical))

    def _extract_matched_canonicals(self, normalized: str) -> list[str]:
        matched: list[str] = []
        for alias in self._aliases:
            if not alias:
                continue
            for _ in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
                canonical = self._aliases_to_canonical[alias]
                if canonical not in matched:
                    matched.append(canonical)
        return matched

    def _extract_matched_mentions(self, original_text: str) -> list[dict[str, str]]:
        mentions: list[dict[str, str]] = []
        seen_spans: set[tuple[int, int]] = set()
        for alias in self._aliases:
            if not alias:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(original_text):
                span = match.span()
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                canonical = self._aliases_to_canonical.get(alias, "")
                raw_text = str(match.group(0)).strip()
                mentions.append(
                    {
                        "raw_text": raw_text,
                        "normalized_key": self._normalize_text(raw_text),
                        "canonical_brand": canonical,
                        "display_name": self.display_name_for(canonical),
                    }
                )
        mentions.sort(key=lambda item: original_text.lower().find(str(item.get("raw_text", "")).lower()))
        deduped: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for item in mentions:
            key = (str(item.get("canonical_brand", "")), str(item.get("raw_text", "")).lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
        return deduped

    def _result_from_raw_mentions(self, raw_mentions: list[str]) -> BrandExtractionResult:
        recognized: list[str] = []
        unknown: list[str] = []
        requested_keys: list[str] = []
        brand_mentions: list[dict[str, object]] = []
        for mention in raw_mentions:
            key = self._normalize_text(mention)
            if key:
                requested_keys.append(key)
            canonical = self._aliases_to_canonical.get(key)
            if canonical:
                if canonical not in recognized:
                    recognized.append(canonical)
                brand_mentions.append(
                    {
                        "raw_text": str(mention).strip(),
                        "normalized_key": key,
                        "recognized": True,
                        "display_name": self.display_name_for(canonical),
                        "canonical_brand": canonical,
                    }
                )
            else:
                display = self._display_surface(mention)
                if display not in unknown:
                    unknown.append(display)
                brand_mentions.append(
                    {
                        "raw_text": str(mention).strip(),
                        "normalized_key": key,
                        "recognized": False,
                        "display_name": display,
                        "canonical_brand": "",
                    }
                )
        return BrandExtractionResult(
            raw_mentions=raw_mentions,
            requested_brand_keys=requested_keys,
            recognized_brands=recognized,
            unknown_brand_mentions=unknown,
            brand_mentions=brand_mentions,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()

    def _extract_raw_mentions(self, original_text: str) -> list[str]:
        candidates = re.split(r"[,\n;/]+|\s+\bи\b\s+|\s+\band\b\s+", original_text, flags=re.IGNORECASE)
        mentions: list[str] = []
        for candidate in candidates:
            token = re.sub(r"[^\w\s-]+", " ", candidate, flags=re.UNICODE)
            token = re.sub(r"\s+", " ", token).strip(" -")
            if not token:
                continue
            if len(token) < 2:
                continue
            normalized_token = self._normalize_text(token)
            if normalized_token in {"подскажи", "подсказывай", "отдельно", "остальные", "что уточнить", "это ты мне уточни"}:
                continue
            mentions.append(token)
        deduped: list[str] = []
        for mention in mentions:
            if mention not in deduped:
                deduped.append(mention)
        return deduped

    def _looks_like_real_brand_list_input(self, original_text: str) -> bool:
        if not self._BRAND_LIST_DELIMITER_RE.search(original_text):
            return False
        if self._NON_BRAND_SIGNAL_RE.search(original_text):
            return False
        candidates = self._extract_raw_mentions(original_text)
        short_candidates = []
        for candidate in candidates:
            normalized = self._normalize_text(candidate)
            if not normalized:
                continue
            if any(char.isdigit() for char in normalized):
                continue
            if len(normalized.split()) > 3:
                continue
            short_candidates.append(candidate)
        return len(short_candidates) >= 2

    @staticmethod
    def _display_surface(surface: str) -> str:
        value = str(surface).strip()
        if not value:
            return ""
        if value.islower() and value.isascii():
            return value.capitalize()
        return value
