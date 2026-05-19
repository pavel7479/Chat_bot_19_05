from __future__ import annotations


class BrandDisplayResolver:
    _DISPLAY_MAP = {
        "uaz": "UAZ",
        "уаз": "UAZ",
        "bmw": "BMW",
        "бмв": "BMW",
        "daf": "DAF",
        "man": "MAN",
        "gac": "GAC",
        "kia": "Kia",
        "киа": "Kia",
        "scania": "Scania",
        "скания": "Scania",
        "haval": "Haval",
        "хавал": "Haval",
        "mitsubishi": "Mitsubishi",
        "митсубиси": "Mitsubishi",
        "мицубиси": "Mitsubishi",
        "mercedes-benz": "Mercedes-Benz",
        "mercedes": "Mercedes",
        "мерседес": "Mercedes",
        "land rover": "Land Rover",
        "toyota": "Toyota",
        "тойота": "Toyota",
    }

    def display(self, canonical_brand: str) -> str:
        key = str(canonical_brand or "").strip().lower()
        if not key:
            return ""
        if key in self._DISPLAY_MAP:
            return self._DISPLAY_MAP[key]
        return " ".join(part.capitalize() for part in key.split())
