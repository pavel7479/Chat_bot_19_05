from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.app.price_provider import PriceProvider
from src.app.product_resolver import ProductResolver
from src.app.slot_extraction_service import SlotExtractionService
from src.domain.brand_display_resolver import BrandDisplayResolver
from src.domain.brands import BrandAliasResolver
from src.domain.pricing import PriceCatalog


class BrandPipelineUnit01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.brands_file = cls.project_root / "src/config/brands.yaml"
        cls.prices_file = cls.project_root / "src/config/prices.yaml"

    def test_brand_alias_resolver_extracts_expected_canonical_brands(self) -> None:
        resolver = BrandAliasResolver(self.brands_file)
        self.assertEqual(resolver.extract("Скания"), ["scania"])
        self.assertEqual(resolver.extract("Scania"), ["scania"])
        self.assertEqual(resolver.extract("УАЗ"), ["uaz"])
        self.assertEqual(resolver.extract("UAZ"), ["uaz"])
        self.assertEqual(resolver.extract("Mitsubishi Outlander"), ["mitsubishi"])
        self.assertEqual(resolver.extract("Haval Dargo"), ["haval"])

    def test_brand_display_resolver_uses_single_display_policy(self) -> None:
        resolver = BrandDisplayResolver()
        self.assertEqual(resolver.display("uaz"), "UAZ")
        self.assertEqual(resolver.display("bmw"), "BMW")
        self.assertEqual(resolver.display("scania"), "Scania")
        self.assertEqual(resolver.display("mercedes-benz"), "Mercedes-Benz")

    def test_slot_extraction_service_extracts_brand_and_requisites(self) -> None:
        service = SlotExtractionService(self.brands_file)
        result = service.extract(
            "Здравствуйте интересует доступ к каталогам Скания на 1 год, для 1 пользователя, ИНН 1234567890, тел. +79117456123, оплата по счёту"
        )
        slots = result.slots
        self.assertEqual(slots.get("brand"), "scania")
        self.assertEqual(slots.get("brand_display"), "Scania")
        self.assertEqual(slots.get("brands"), ["scania"])
        self.assertEqual(slots.get("phone"), "+79117456123")
        self.assertEqual(slots.get("inn"), "1234567890")
        self.assertEqual(slots.get("period"), "12_months")
        self.assertEqual(slots.get("user_count"), 1)
        self.assertEqual(slots.get("payment_method"), "invoice")
        self.assertEqual(slots.get("raw_brand_mentions"), ["Скания"])
        self.assertEqual(slots.get("requested_brand_keys"), ["скания"])
        self.assertNotIn("unknown_brand_mentions", slots)

    def test_brand_alias_resolver_extracts_brand_from_long_requisites_message(self) -> None:
        resolver = BrandAliasResolver(self.brands_file)
        result = resolver.extract_detailed(
            "Здравствуйте интересует доступ к каталогам Скания на 1 год, для 1 пользователя, ИНН 1234567890, тел. +79117456123, оплата по счёту"
        )

        self.assertEqual(result.recognized_brands, ["scania"])
        self.assertEqual(result.raw_mentions, ["Скания"])
        self.assertEqual(result.requested_brand_keys, ["скания"])
        self.assertEqual(result.unknown_brand_mentions, [])

    def test_brand_alias_resolver_keeps_brand_list_mode_for_mixed_brand_list(self) -> None:
        resolver = BrandAliasResolver(self.brands_file)
        result = resolver.extract_detailed("mers, ford, автоваз, Captiva и РСД-10")

        self.assertEqual(result.recognized_brands, ["ford", "lada"])
        self.assertEqual(result.unknown_brand_mentions, ["Mers", "Captiva", "РСД-10"])

    def test_slot_extraction_service_extracts_brand_inside_phrase_without_commas(self) -> None:
        service = SlotExtractionService(self.brands_file)
        result = service.extract("нужен доступ к каталогам Скания на год")

        self.assertEqual(result.slots.get("brand"), "scania")
        self.assertEqual(result.slots.get("brand_display"), "Scania")

    def test_price_provider_returns_tis_price_when_found(self) -> None:
        provider = PriceProvider(price_catalog=PriceCatalog(self.prices_file))
        context = provider.build(required_price_blocks=["tis"], normalized_brands=["mitsubishi"])
        self.assertEqual(context.tis_price_status, "found")
        self.assertTrue(any("mitsubishi" in line.lower() for line in context.price_lines))

    def test_price_provider_returns_epc_fallback_when_tis_price_missing(self) -> None:
        provider = PriceProvider(price_catalog=PriceCatalog(self.prices_file))
        context = provider.build(required_price_blocks=["tis"], normalized_brands=["haval"])
        self.assertEqual(context.tis_price_status, "missing")
        self.assertEqual(context.missing_tis_price_brands, ["haval"])
        self.assertIn("epc", context.fallback_price_blocks)
        self.assertTrue(any("прайсе не указана" in line.lower() for line in context.price_lines))
        self.assertTrue(any("epc full" in line.lower() for line in context.price_lines))

    def test_product_resolver_separates_brand_tis_partial_and_vag(self) -> None:
        resolver = ProductResolver(self.brands_file)

        brand_case = resolver.resolve(user_query="только УАЗ")
        self.assertEqual(brand_case.mentioned_brands, ["uaz"])
        self.assertEqual(brand_case.inferred_product_context, "tis")
        self.assertFalse(brand_case.partial_access_requested)

        partial_case = resolver.resolve(user_query="можно купить только один бренд в EPC?")
        self.assertTrue(partial_case.partial_access_requested)
        self.assertEqual(partial_case.mentioned_brands, [])

        vag_case = resolver.resolve(user_query="только VAG")
        self.assertEqual(vag_case.unsupported_brand_group, "VAG")
        self.assertEqual(vag_case.mentioned_brands, [])


if __name__ == "__main__":
    unittest.main()
