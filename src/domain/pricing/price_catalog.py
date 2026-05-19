from __future__ import annotations

from pathlib import Path

import yaml

from src.domain.brand_display_resolver import BrandDisplayResolver


class PriceCatalog:
    def __init__(self, prices_file_path: Path) -> None:
        raw = self._load_yaml(prices_file_path)
        self._epc_prices = self._load_epc_prices(raw)
        self._tis_prices = self._load_tis_prices(raw)
        self._brand_display = BrandDisplayResolver()

    @staticmethod
    def _load_yaml(prices_file_path: Path) -> dict[str, object]:
        if not prices_file_path.exists():
            return {}
        raw = yaml.safe_load(prices_file_path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _load_epc_prices(raw: dict[str, object]) -> dict[str, int]:
        epc_raw = raw.get("epc_prices", {})
        if not isinstance(epc_raw, dict):
            return {}
        prices: dict[str, int] = {}
        for key, value in epc_raw.items():
            name = str(key).strip().lower()
            if not name:
                continue
            try:
                prices[name] = int(value)
            except (TypeError, ValueError):
                continue
        return prices

    @staticmethod
    def _load_tis_prices(raw: dict[str, object]) -> dict[str, int]:
        tis_raw = raw.get("tis_prices", {})
        if not isinstance(tis_raw, dict):
            return {}
        prices: dict[str, int] = {}
        for key, value in tis_raw.items():
            name = str(key).strip().lower()
            if not name:
                continue
            try:
                prices[name] = int(value)
            except (TypeError, ValueError):
                continue
        return prices

    def has_epc_period(self, period_key: str) -> bool:
        return str(period_key).strip().lower() in self._epc_prices

    def epc_price_for(self, period_key: str) -> int:
        return self._epc_prices[str(period_key).strip().lower()]

    def epc_prices(self) -> dict[str, int]:
        return dict(self._epc_prices)

    def has_tis_price(self, canonical_brand: str) -> bool:
        return str(canonical_brand).strip().lower() in self._tis_prices

    def tis_price_for(self, canonical_brand: str) -> int:
        return self._tis_prices[str(canonical_brand).strip().lower()]

    def tis_prices(self) -> dict[str, int]:
        return dict(self._tis_prices)

    def display_name(self, canonical_brand: str) -> str:
        return self._brand_display.display(canonical_brand)
