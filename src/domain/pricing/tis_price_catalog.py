from __future__ import annotations

from pathlib import Path

from .price_catalog import PriceCatalog


class TisPriceCatalog:
    def __init__(self, prices_file_path: Path) -> None:
        self._catalog = PriceCatalog(prices_file_path)

    def has_price(self, canonical_brand: str) -> bool:
        return self._catalog.has_tis_price(canonical_brand)

    def price_for(self, canonical_brand: str) -> int:
        return self._catalog.tis_price_for(canonical_brand)

    def display_name(self, canonical_brand: str) -> str:
        return self._catalog.display_name(canonical_brand)
