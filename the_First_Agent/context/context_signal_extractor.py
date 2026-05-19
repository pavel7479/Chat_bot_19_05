from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

from src.core.models import ContextSignals, SessionState


@dataclass(slots=True)
class _SemanticRule:
    id: str
    when_meaning_contains_any: tuple[str, ...] = ()
    when_gist_contains_any: tuple[str, ...] = ()
    when_user_query_contains_any: tuple[str, ...] = ()
    when_last_assistant_contains_any: tuple[str, ...] = ()
    when_query_kind_in: tuple[str, ...] = ()
    semantic_flags: tuple[str, ...] = ()
    boost_topics: tuple[str, ...] = ()
    penalty_topics: tuple[str, ...] = ()
    continuity_topics: tuple[str, ...] = ()


@dataclass(slots=True)
class _SemanticRoutingMaps:
    boost_topics: set[str] = field(default_factory=set)
    penalty_topics: set[str] = field(default_factory=set)
    continuity_topics: set[str] = field(default_factory=set)
    semantic_flags: set[str] = field(default_factory=set)
    matched_rules: list[str] = field(default_factory=list)


class ContextSignalExtractor:
    _TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
    _SHORT_REPLY_EXACT = {
        "да", "нет", "являюсь", "все", "всё", "ага", "ок", "yes", "no", "неа", "nope", "yep",
        "а какие есть", "какие есть", "на него", "для него", "по нему",
    }
    _SHORT_REPLY_TOKENS = {
        "да", "нет", "являюсь", "все", "всё", "ага", "ок", "yes", "no", "nope", "yep",
        "юрлиц", "legal", "физлицо", "него", "него", "нему",
    }
    _CONTINUITY_EXCLUDED_TOPICS = {"out_of_scope_request", "nonsense_input"}

    def __init__(self, rules_path: Path) -> None:
        self._rules_path = rules_path
        self._rules = self._load_rules(rules_path)

    def extract(
        self,
        *,
        user_query: str,
        meaning: str,
        gist: str,
        history_text: str,
        session_state: SessionState | None,
        fallback_used: bool,
    ) -> ContextSignals:
        signals = ContextSignals(
            user_query=str(user_query or ""),
            gist=str(gist or ""),
            meaning=str(meaning or ""),
            fallback_used=bool(fallback_used),
        )
        if signals.fallback_used:
            return signals

        query_kind = self._detect_query_kind(signals.user_query)
        last_assistant = self._extract_last_assistant_message(history_text)
        routing = self._match_rules(
            meaning=self._normalize_text(signals.meaning),
            gist=self._normalize_text(signals.gist),
            user_query=self._normalize_text(signals.user_query),
            last_assistant=self._normalize_text(last_assistant),
            query_kind=query_kind,
        )
        continuity_topics = set(routing.continuity_topics)
        continuity_topics.update(self._derive_continuity_topics(query_kind, last_assistant, session_state))

        signals.semantic_flags.update(routing.semantic_flags)
        signals.semantic_boost_topics.update(routing.boost_topics)
        signals.semantic_penalty_topics.update(routing.penalty_topics)
        signals.continuity_topics.update(continuity_topics)
        return signals

    def build_trace(self, context_signals: ContextSignals) -> dict[str, object]:
        return {
            "semantic_flags": sorted(context_signals.semantic_flags),
            "boost_topics": sorted(context_signals.semantic_boost_topics),
            "penalty_topics": sorted(context_signals.semantic_penalty_topics),
            "continuity_topics": sorted(context_signals.continuity_topics),
            "fallback_used": bool(context_signals.fallback_used),
        }

    def _match_rules(
        self,
        *,
        meaning: str,
        gist: str,
        user_query: str,
        last_assistant: str,
        query_kind: str,
    ) -> _SemanticRoutingMaps:
        routing = _SemanticRoutingMaps()
        for rule in self._rules:
            if rule.when_meaning_contains_any and not self._contains_any(meaning, rule.when_meaning_contains_any):
                continue
            if rule.when_gist_contains_any and not self._contains_any(gist, rule.when_gist_contains_any):
                continue
            if rule.when_user_query_contains_any and not self._contains_any(user_query, rule.when_user_query_contains_any):
                continue
            if rule.when_last_assistant_contains_any and not self._contains_any(last_assistant, rule.when_last_assistant_contains_any):
                continue
            if rule.when_query_kind_in and query_kind not in rule.when_query_kind_in:
                continue
            routing.matched_rules.append(rule.id)
            routing.semantic_flags.update(rule.semantic_flags)
            routing.boost_topics.update(rule.boost_topics)
            routing.penalty_topics.update(rule.penalty_topics)
            routing.continuity_topics.update(rule.continuity_topics)
        return routing

    def _derive_continuity_topics(
        self,
        query_kind: str,
        last_assistant: str,
        session_state: SessionState | None,
    ) -> set[str]:
        if query_kind not in {"short_reply", "pronoun_reply", "ellipsis_reply", "confirmation", "denial"}:
            return set()
        topics: set[str] = set()
        if session_state is not None:
            topics.update(
                topic_id
                for topic_id in session_state.last_topic_ids
                if topic_id and topic_id not in self._CONTINUITY_EXCLUDED_TOPICS
            )
        text = self._normalize_text(last_assistant)
        if text:
            if re.search(r"\b(бренд|марк|каталог)\b", text):
                topics.add("brand_list_request")
            if re.search(r"\b(демо|demo|trial|пробн|тестов)\b", text):
                topics.add("demo_access")
            if re.search(r"\b(api|интеграц|1с|crm)\b", text):
                topics.add("api_integration")
        return topics

    @classmethod
    def _extract_last_assistant_message(cls, history_text: str) -> str:
        lines = [line.strip() for line in str(history_text).splitlines() if line.strip()]
        for line in reversed(lines):
            if line.lower().startswith(("assistant:", "бот:")):
                return line.split(":", 1)[1].strip()
        return ""

    @classmethod
    def _detect_query_kind(cls, user_query: str) -> str:
        normalized = cls._normalize_text(user_query)
        if not normalized:
            return "ellipsis_reply"
        if normalized in {".", "..", "...", "...."}:
            return "ellipsis_reply"
        if normalized in {"да", "yes", "yep", "ага"}:
            return "confirmation"
        if normalized in {"нет", "no", "nope", "неа"}:
            return "denial"
        if normalized in {"на него", "для него", "по нему"}:
            return "pronoun_reply"
        tokens = set(cls._tokenize(normalized))
        if normalized in cls._SHORT_REPLY_EXACT:
            return "short_reply"
        if tokens and len(tokens) <= 2 and tokens <= cls._SHORT_REPLY_TOKENS:
            return "short_reply"
        return "full_query"

    @classmethod
    def _contains_any(cls, text: str, fragments: tuple[str, ...]) -> bool:
        return any(cls._normalize_text(fragment) in text for fragment in fragments if fragment)

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [match.group(0) for match in cls._TOKEN_RE.finditer(cls._normalize_text(text))]

    @classmethod
    def _load_rules(cls, path: Path) -> list[_SemanticRule]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules: list[_SemanticRule] = []
        for item in raw.get("rules", []):
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("id", "")).strip()
            if not rule_id:
                continue
            rules.append(
                _SemanticRule(
                    id=rule_id,
                    when_meaning_contains_any=tuple(str(x).strip() for x in item.get("when_meaning_contains_any", []) if str(x).strip()),
                    when_gist_contains_any=tuple(str(x).strip() for x in item.get("when_gist_contains_any", []) if str(x).strip()),
                    when_user_query_contains_any=tuple(str(x).strip() for x in item.get("when_user_query_contains_any", []) if str(x).strip()),
                    when_last_assistant_contains_any=tuple(str(x).strip() for x in item.get("when_last_assistant_contains_any", []) if str(x).strip()),
                    when_query_kind_in=tuple(str(x).strip() for x in item.get("when_query_kind_in", []) if str(x).strip()),
                    semantic_flags=tuple(str(x).strip() for x in item.get("semantic_flags", []) if str(x).strip()),
                    boost_topics=tuple(str(x).strip() for x in item.get("boost_topics", []) if str(x).strip()),
                    penalty_topics=tuple(str(x).strip() for x in item.get("penalty_topics", []) if str(x).strip()),
                    continuity_topics=tuple(str(x).strip() for x in item.get("continuity_topics", []) if str(x).strip()),
                )
            )
        return rules
