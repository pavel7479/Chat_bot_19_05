from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class RuleEngine:
    """Declarative regex rule engine for response understanding/focus."""

    def __init__(self, rules_path: Path) -> None:
        raw = yaml.safe_load(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else {}
        self._rules = raw if isinstance(raw, dict) else {}
        patterns = self._rules.get("patterns", {})
        self._patterns: dict[str, re.Pattern[str]] = {}
        if isinstance(patterns, dict):
            for name, pattern in patterns.items():
                if not isinstance(pattern, str) or not pattern.strip():
                    continue
                self._patterns[str(name).strip()] = re.compile(pattern)

    def match(self, name: str, text: str) -> bool:
        pattern = self._patterns.get(name)
        if pattern is None:
            return False
        return bool(pattern.search(text))

    def findall_count(self, name: str, text: str) -> int:
        pattern = self._patterns.get(name)
        if pattern is None:
            return 0
        return len(pattern.findall(text))

    def match_multiline(self, name: str, text: str) -> bool:
        pattern = self._patterns.get(name)
        if pattern is None:
            return False
        return bool(re.search(pattern.pattern, text, flags=re.MULTILINE))

    def resolve_focus(
        self,
        query: str,
        topic_set: set[str],
        primary_topic: str,
        price_query: bool,
        compare_epc_tis_query: bool,
    ) -> str:
        rules = self._rules.get("focus_rules", [])
        if isinstance(rules, list):
            for item in rules:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("pattern", "")).strip()
                required_any = {str(v).strip() for v in item.get("required_topics_any", []) if str(v).strip()}
                required_all = {str(v).strip() for v in item.get("required_topics_all", []) if str(v).strip()}
                excluded = {str(v).strip() for v in item.get("excluded_topics", []) if str(v).strip()}
                need_price = item.get("require_price_query")
                need_compare = item.get("require_compare_query")
                if required_all and not required_all.issubset(topic_set):
                    continue
                if required_any and not topic_set.intersection(required_any):
                    continue
                if excluded and topic_set.intersection(excluded):
                    continue
                if need_price is True and not price_query:
                    continue
                if need_price is False and price_query:
                    continue
                if need_compare is True and not compare_epc_tis_query:
                    continue
                if need_compare is False and compare_epc_tis_query:
                    continue
                if name and not self.match(name, query):
                    continue
                action = str(item.get("focus", "")).strip()
                if action:
                    return action
        both_tariff_topics = "epc_tariffs" in topic_set and "tis_tariffs" in topic_set
        tis_mentioned = self.match("tis_mentioned", query)
        epc_mentioned = self.match("epc_mentioned", query)
        if compare_epc_tis_query and (both_tariff_topics or (tis_mentioned and epc_mentioned)):
            return "compare_epc_tis"
        if tis_mentioned and not epc_mentioned:
            return "tis_tariffs"
        if epc_mentioned and not tis_mentioned:
            return "epc_tariffs"
        if tis_mentioned and epc_mentioned:
            return "compare_epc_tis"
        if not price_query and not both_tariff_topics and primary_topic not in {"tis_tariffs", "epc_tariffs"}:
            return "unknown"
        if both_tariff_topics and primary_topic in {"tis_tariffs", "epc_tariffs"}:
            return primary_topic
        if "tis_tariffs" in topic_set and "epc_tariffs" not in topic_set:
            return "tis_tariffs"
        if "epc_tariffs" in topic_set and "tis_tariffs" not in topic_set:
            return "epc_tariffs"
        if "company_services_info" in topic_set:
            return "company_services"
        return "unknown"
