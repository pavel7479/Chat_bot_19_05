from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class IntentConfigLoader:
    def __init__(self, intents_path: Path, brands_path: Path) -> None:
        self._intents_path = intents_path
        self._brands_path = brands_path

    def load(self) -> dict[str, Any]:
        intents_raw = yaml.safe_load(self._intents_path.read_text(encoding="utf-8")) or {}
        brands_raw = yaml.safe_load(self._brands_path.read_text(encoding="utf-8")) or {}
        intents_raw.setdefault("signals", {})
        intents_raw["signals"]["brand_aliases"] = self._extract_brand_aliases(brands_raw)
        return intents_raw

    @staticmethod
    def _extract_brand_aliases(brands_raw: dict[str, Any]) -> list[str]:
        aliases: list[str] = []
        for item in brands_raw.get("brands", []):
            canonical = str(item.get("canonical", "")).strip()
            if canonical:
                aliases.append(canonical.lower())
            for alias in item.get("aliases", []):
                norm = str(alias).strip().lower()
                if norm:
                    aliases.append(norm)
        deduped: list[str] = []
        for alias in aliases:
            if alias not in deduped:
                deduped.append(alias)
        return deduped
