from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from src.core.models import ContextSignals, SessionState, TopicDefinition


@dataclass(slots=True)
class TopicShortlistCandidate:
    topic_id: str
    label_ru: str
    score: float
    lexical_score: float
    semantic_boost: float
    semantic_penalty: float
    continuity_boost: float
    overlap: float
    jaccard: float
    avoid_overlap: float
    matched_tokens: list[str]
    score_breakdown: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "label_ru": self.label_ru,
            "score": self.score,
            "lexical_score": self.lexical_score,
            "semantic_boost": self.semantic_boost,
            "semantic_penalty": self.semantic_penalty,
            "continuity_boost": self.continuity_boost,
            "overlap": self.overlap,
            "jaccard": self.jaccard,
            "avoid_overlap": self.avoid_overlap,
            "matched_tokens": list(self.matched_tokens),
            "score_breakdown": dict(self.score_breakdown),
        }


@dataclass(slots=True)
class _SourceScore:
    score: float
    overlap: float
    jaccard: float
    avoid_overlap: float
    matched_tokens: set[str]


@dataclass(slots=True)
class _TopicSignals:
    positive_tokens: set[str]
    anti_tokens: set[str]


class TopicShortlistBuilder:
    _TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
    _STOPWORDS = {
        "и", "а", "но", "или", "же", "ли", "на", "в", "во", "по", "к", "ко", "с", "со", "у", "о", "об", "от",
        "до", "из", "за", "для", "это", "как", "что", "то", "бы", "ну", "мне", "мы", "вы", "я", "он", "она",
        "они", "нас", "вам", "их", "его", "ее", "еще", "ещё", "тогда", "просто", "вообще", "либо", "если",
        "был", "была", "были", "будет", "есть", "нужен", "нужна", "нужно", "нужны", "могу", "можно",
        "ваш", "ваша", "ваши", "наш", "наша", "наши", "клиент", "реплика", "последняя", "суть", "смысл",
        "спрашивает", "уточняет", "запрашивает", "сообщает", "говорит", "продукт", "ты",
    }
    _KEEP_TOKENS = {"нет", "без", "да", "demo", "tis", "epc"}
    _QUERY_WEIGHT = 1.0
    _MEANING_WEIGHT = 0.8
    _BRAND_BOOST = 1.35
    _BRAND_LIST_BOOST = 1.05
    _TIS_EXACT_BOOST = 1.05
    _TIS_BRAND_CONTEXT_BOOST = 0.9
    _RELATION_BOOST = 1.15
    _SHORT_REPLY_STATUS_BOOST = 1.2
    _SHORT_REPLY_DEMO_BOOST = 0.85
    _ABUSE_BOOST = 1.1
    _NOISE_BOOST = 1.25
    _GENERAL_PRICING_BOOST = 0.92
    _MULTI_ACCESS_BOOST = 1.15
    _FUZZY_BRAND_THRESHOLD = 0.78
    _ABUSE_PATTERNS = (
        re.compile(r"\bмошенн(ик|ики|ичество)\b"),
        re.compile(r"\bтуп(ишь|ой|ые|ая)\b"),
        re.compile(r"\bразвод\b"),
        re.compile(r"\bидиот\b"),
        re.compile(r"\bбред\b"),
    )
    _TOKEN_NORMALIZATIONS = {
        "стоимость": "цена",
        "тариф": "цена",
        "тарифы": "цена",
        "сколько": "цена",
        "стоит": "цена",
        "почем": "цена",
        "епс": "epc",
        "тис": "tis",
        "счёт": "счет",
        "счета": "счет",
        "счёта": "счет",
        "физик": "физлицо",
        "физ": "физлицо",
        "физлица": "физлицо",
        "физлицом": "физлицо",
        "физлицами": "физлицо",
        "юрлица": "юрлиц",
        "юрлицо": "юрлиц",
        "юрлицам": "юрлиц",
        "юрлицами": "юрлиц",
        "юридическое": "юрлиц",
        "юридическим": "юрлиц",
        "юридическими": "юрлиц",
        "юридического": "юрлиц",
        "yes": "да",
        "yep": "да",
        "no": "нет",
        "nope": "нет",
        "ага": "да",
        "неа": "нет",
        "legal": "юрлиц",
        "мерседес": "mercedes",
        "мерс": "mercedes",
        "фольксваген": "volkswagen",
        "фольцваген": "volkswagen",
        "вольцваген": "volkswagen",
        "тойота": "toyota",
        "тайота": "toyota",
        "лексус": "lexus",
        "порше": "porsche",
        "пежо": "peugeot",
        "рено": "renault",
        "хундай": "hyundai",
        "хендай": "hyundai",
        "кия": "kia",
        "киа": "kia",
        "бмв": "bmw",
        "вольво": "volvo",
        "ауди": "audi",
        "отличие": "разница",
        "отличается": "разница",
        "отличаются": "разница",
        "различие": "разница",
        "входит": "входит",
        "входят": "входит",
        "включает": "входит",
        "включено": "входит",
        "отдельно": "отдельно",
        "отдельный": "отдельно",
        "отдельные": "отдельно",
        "вместе": "вместе",
        "состав": "состав",
    }
    _BRAND_LIST_PATTERNS = (
        re.compile(r"\bкакие бренды\b"),
        re.compile(r"\bкакие марки\b"),
        re.compile(r"\bсписок брендов\b"),
        re.compile(r"\bсписок марок\b"),
        re.compile(r"\bа какие есть\b"),
        re.compile(r"^все$"),
    )
    _CATALOG_LIST_PATTERNS = (
        re.compile(r"\bкакие каталоги\b"),
        re.compile(r"\bкакие продукты\b"),
        re.compile(r"\bкакие типы каталогов\b"),
        re.compile(r"\bчто у вас есть\b"),
        re.compile(r"\bчто есть кроме epc\b"),
        re.compile(r"\bчто есть кроме tis\b"),
    )
    _PARTIAL_CATALOG_PATTERNS = (
        re.compile(r"\bтолько одного бренда\b"),
        re.compile(r"\bпредоставить каталоги только одного бренда\b"),
        re.compile(r"\bодин бренд в (?:epc|епс)\b"),
        re.compile(r"\bне полный пакет\b"),
        re.compile(r"\bчасть каталога\b"),
        re.compile(r"\bотдельно по брендам\b"),
    )
    _SAFE_FALLBACK_IDS = [
        "nonsense_input",
        "out_of_scope_request",
    ]

    def __init__(
        self,
        topics: dict[str, TopicDefinition],
        *,
        top_k: int = 8,
        out_of_scope_threshold: float = 0.32,
        brands_file_path: Path | None = None,
        facts_file_path: Path | None = None,
    ) -> None:
        del facts_file_path
        self._topics = dict(topics)
        self._top_k = max(int(top_k), 2)
        self._out_of_scope_threshold = float(out_of_scope_threshold)
        project_root = Path(__file__).resolve().parents[2]
        self._brands_file_path = brands_file_path or (project_root / "src/config/brands.yaml")
        self._brand_aliases = self._load_brand_aliases(self._brands_file_path)
        self._topic_signals = {
            topic_id: self._build_topic_signals(topic)
            for topic_id, topic in self._topics.items()
        }
        self._last_full_shortlist_scores: list[dict[str, object]] = []
        self._last_selected_shortlist_scores: list[dict[str, object]] = []
        self._last_semantic_routing_trace: dict[str, object] = {
            "meaning_used": False,
            "brand_alias_hits": [],
            "brand_fuzzy_hits": [],
            "tis_boost_active": False,
            "product_relation_boost_active": False,
            "abuse_boost_active": False,
            "noise_boost_active": False,
            "general_pricing_boost_active": False,
            "multi_access_boost_active": False,
            "zero_score_fallback_used": False,
            "fallback_used": False,
        }

    def build_shortlist(
        self,
        query: str,
        history_text: str = "",
        session_state: SessionState | None = None,
        context_signals: ContextSignals | None = None,
        top_k: int | None = None,
    ) -> list[TopicShortlistCandidate]:
        del history_text, session_state
        limit = max(int(top_k or self._top_k), 2)
        raw_query = str(query or "")
        normalized_query = self._normalize_text(raw_query)
        query_tokens = self._build_query_tokens(raw_query)
        meaning_tokens = self._build_meaning_tokens(context_signals)
        alias_hits, fuzzy_hits = self._detect_brand_hits(raw_query)
        brand_hits = alias_hits | fuzzy_hits
        query_boosts = self._build_query_boosts(raw_query, normalized_query, brand_hits, context_signals)

        candidates = []
        for topic_id, topic in self._topics.items():
            candidate = self._score_topic(
                topic=topic,
                signals=self._topic_signals[topic_id],
                query_tokens=query_tokens,
                meaning_tokens=meaning_tokens,
                brand_hits=brand_hits,
                query_boosts=query_boosts,
            )
            candidates.append(candidate)

        ranked = sorted(
            candidates,
            key=lambda item: (-item.score, -item.overlap, -item.jaccard, item.topic_id),
        )
        zero_score_fallback_used = max((item.score for item in ranked), default=0.0) <= 0.0
        selected = self._safe_fallback_shortlist(ranked, brand_hits, raw_query, limit) if zero_score_fallback_used else ranked[:limit]
        selected = self._ensure_required_topics(
            selected=selected,
            ranked=ranked,
            normalized_query=normalized_query,
            limit=limit,
        )
        self._last_full_shortlist_scores = [item.as_dict() for item in ranked]
        self._last_selected_shortlist_scores = [item.as_dict() for item in selected]

        self._last_semantic_routing_trace = {
            "meaning_used": bool(meaning_tokens),
            "brand_alias_hits": sorted(alias_hits),
            "brand_fuzzy_hits": sorted(fuzzy_hits),
            "tis_boost_active": bool(query_boosts.get("tis_tariffs")),
            "product_relation_boost_active": bool(query_boosts.get("product_relation_or_difference")),
            "abuse_boost_active": bool(query_boosts.get("human_operator_request")),
            "noise_boost_active": bool(query_boosts.get("nonsense_input")),
            "general_pricing_boost_active": bool(query_boosts.get("epc_tariffs")) and bool(query_boosts.get("tis_tariffs")),
            "multi_access_boost_active": bool(query_boosts.get("multi_device_access")),
            "zero_score_fallback_used": zero_score_fallback_used,
            "fallback_used": bool(context_signals.fallback_used) if context_signals else False,
        }
        return selected

    def build_topic_ids(
        self,
        query: str,
        history_text: str = "",
        session_state: SessionState | None = None,
        context_signals: ContextSignals | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        return [
            item.topic_id
            for item in self.build_shortlist(
                query,
                history_text,
                session_state,
                context_signals,
                top_k,
            )
        ]

    def get_last_semantic_routing_trace(self) -> dict[str, object]:
        return dict(self._last_semantic_routing_trace)

    def get_last_full_shortlist_scores(self) -> list[dict[str, object]]:
        return list(self._last_full_shortlist_scores)

    def get_last_selected_shortlist_scores(self) -> list[dict[str, object]]:
        return list(self._last_selected_shortlist_scores)

    def _score_topic(
        self,
        *,
        topic: TopicDefinition,
        signals: _TopicSignals,
        query_tokens: set[str],
        meaning_tokens: set[str],
        brand_hits: set[str],
        query_boosts: dict[str, float],
    ) -> TopicShortlistCandidate:
        query_score = self._score_source_tokens(query_tokens, signals)
        meaning_score = self._score_source_tokens(meaning_tokens, signals)
        lexical_score = (query_score.score * self._QUERY_WEIGHT) + (meaning_score.score * self._MEANING_WEIGHT)
        brand_boost = self._brand_boost(topic.id, brand_hits)
        query_boost = float(query_boosts.get(topic.id, 0.0))
        final_score = lexical_score + brand_boost + query_boost
        matched_tokens = (query_score.matched_tokens | meaning_score.matched_tokens) - self._STOPWORDS
        overlap = max(query_score.overlap, meaning_score.overlap)
        jaccard = max(query_score.jaccard, meaning_score.jaccard)
        avoid_overlap = max(query_score.avoid_overlap, meaning_score.avoid_overlap)

        return TopicShortlistCandidate(
            topic_id=topic.id,
            label_ru=topic.title,
            score=round(final_score, 6),
            lexical_score=round(lexical_score, 6),
            semantic_boost=round(brand_boost + query_boost, 6),
            semantic_penalty=0.0,
            continuity_boost=0.0,
            overlap=round(overlap, 6),
            jaccard=round(jaccard, 6),
            avoid_overlap=round(avoid_overlap, 6),
            matched_tokens=sorted(matched_tokens),
            score_breakdown={
                "query": round(query_score.score, 6),
                "meaning": round(meaning_score.score, 6),
                "brand_boost": round(brand_boost, 6),
                "query_boost": round(query_boost, 6),
                "final": round(final_score, 6),
            },
        )

    def _brand_boost(self, topic_id: str, brand_hits: set[str]) -> float:
        if topic_id == "specific_brand_check" and brand_hits:
            return self._BRAND_BOOST
        return 0.0

    def _build_query_boosts(
        self,
        raw_query: str,
        normalized_query: str,
        brand_hits: set[str],
        context_signals: ContextSignals | None,
    ) -> dict[str, float]:
        boosts: dict[str, float] = {}
        has_explicit_tis = self._contains_explicit_tis_signal(normalized_query)
        has_explicit_epc = self._contains_explicit_epc_signal(normalized_query)
        if has_explicit_tis:
            boosts["tis_tariffs"] = self._TIS_EXACT_BOOST
        if self._contains_general_pricing_signal(normalized_query) and not has_explicit_tis and not has_explicit_epc:
            boosts["epc_tariffs"] = max(boosts.get("epc_tariffs", 0.0), self._GENERAL_PRICING_BOOST)
            boosts["tis_tariffs"] = max(boosts.get("tis_tariffs", 0.0), self._GENERAL_PRICING_BOOST)
        if self._is_brand_list_request(normalized_query, brand_hits):
            boosts["brand_list_request"] = self._BRAND_LIST_BOOST
        if self._is_catalog_list_request(normalized_query, brand_hits):
            boosts["catalog_list_request"] = self._BRAND_LIST_BOOST
        if self._is_partial_catalog_request(normalized_query, brand_hits):
            boosts["partial_catalog_request"] = max(boosts.get("partial_catalog_request", 0.0), 1.28)
            boosts["specific_brand_check"] = min(boosts.get("specific_brand_check", 0.0), -0.42)
        if brand_hits and self._is_tis_brand_context(normalized_query, context_signals):
            boosts["tis_tariffs"] = max(boosts.get("tis_tariffs", 0.0), self._TIS_BRAND_CONTEXT_BOOST)
        if self._contains_product_relation_signal(normalized_query, context_signals):
            boosts["product_relation_or_difference"] = self._RELATION_BOOST
        if self._contains_multi_access_signal(normalized_query):
            boosts["multi_device_access"] = self._MULTI_ACCESS_BOOST
        if re.search(r"\bсамозанят\w*\b", normalized_query):
            boosts["self_employed_purchase"] = max(boosts.get("self_employed_purchase", 0.0), 1.18)
        if re.search(r"\b(договор|номер договора|существующий договор|проверить договор)\b", normalized_query):
            boosts["existing_contract_check"] = max(boosts.get("existing_contract_check", 0.0), 1.18)
        if re.search(r"\b(как оплатить|способ(?:ы)? оплаты|по счету|по qr|картой|по карте)\b", normalized_query):
            boosts["payment_process"] = max(boosts.get("payment_process", 0.0), 1.2)
        if re.search(r"\b(как продлить|продление подписки|продлить доступ)\b", normalized_query):
            boosts["subscription_renewal"] = max(boosts.get("subscription_renewal", 0.0), 1.2)
        if re.search(r"\bконкурент\w*\b", normalized_query):
            boosts["competitor_comparison"] = max(boosts.get("competitor_comparison", 0.0), 1.1)
        if context_signals is not None:
            for topic_id in context_signals.semantic_boost_topics:
                boosts[topic_id] = max(boosts.get(topic_id, 0.0), 0.88)
            for topic_id in context_signals.continuity_topics:
                boosts[topic_id] = max(boosts.get(topic_id, 0.0), 0.46)
            for topic_id in context_signals.semantic_penalty_topics:
                boosts[topic_id] = min(boosts.get(topic_id, 0.0), -0.55)
        boosts.update(self._build_short_reply_boosts(normalized_query, context_signals))
        if self._contains_abuse(normalized_query):
            boosts["human_operator_request"] = self._ABUSE_BOOST
        if self._is_noise_like(raw_query):
            boosts["nonsense_input"] = self._NOISE_BOOST
        if brand_hits:
            boosts.setdefault("specific_brand_check", 0.0)
        return boosts

    def _ensure_required_topics(
        self,
        *,
        selected: list[TopicShortlistCandidate],
        ranked: list[TopicShortlistCandidate],
        normalized_query: str,
        limit: int,
    ) -> list[TopicShortlistCandidate]:
        candidate_map = {item.topic_id: item for item in ranked}
        selected_ids = [item.topic_id for item in selected]
        required_ids: list[str] = []

        if self._contains_general_pricing_signal(normalized_query):
            if "purchase_ready" in candidate_map:
                required_ids.append("purchase_ready")
            if not self._contains_explicit_epc_signal(normalized_query) and not self._contains_explicit_tis_signal(normalized_query):
                for topic_id in ("epc_tariffs", "tis_tariffs"):
                    if topic_id in candidate_map:
                        required_ids.append(topic_id)

        if self._contains_multi_access_signal(normalized_query) and "multi_device_access" in candidate_map:
            required_ids.append("multi_device_access")
        if self._is_catalog_list_request(normalized_query, set()) and "catalog_list_request" in candidate_map:
            required_ids.append("catalog_list_request")
        if self._is_partial_catalog_request(normalized_query, set()) and "partial_catalog_request" in candidate_map:
            required_ids.append("partial_catalog_request")

        for topic_id in required_ids:
            if topic_id in selected_ids:
                continue
            if len(selected) < limit:
                selected.append(candidate_map[topic_id])
                selected_ids.append(topic_id)
                continue
            selected[-1] = candidate_map[topic_id]
            selected_ids[-1] = topic_id

        unique: list[TopicShortlistCandidate] = []
        seen: set[str] = set()
        for item in selected:
            if item.topic_id in seen:
                continue
            seen.add(item.topic_id)
            unique.append(item)
        return unique[:limit]

    def _contains_product_relation_signal(
        self,
        normalized_query: str,
        context_signals: ContextSignals | None,
    ) -> bool:
        combined_text = normalized_query
        if context_signals is not None and not context_signals.fallback_used:
            combined_text = f"{combined_text} {context_signals.meaning}".strip()
        tokens = set(self._tokenize(combined_text))
        if not tokens:
            return False
        product_tokens = {"epc", "tis"}
        relation_tokens = {"входит", "отдельно", "разница", "вместе", "состав"}
        return bool(tokens & product_tokens) and bool(tokens & relation_tokens)

    @staticmethod
    def _contains_explicit_tis_signal(normalized_query: str) -> bool:
        return bool(re.search(r"\b(tis|тис)\b", normalized_query))

    @staticmethod
    def _contains_explicit_epc_signal(normalized_query: str) -> bool:
        return bool(re.search(r"\b(epc|епс)\b", normalized_query))

    def _contains_general_pricing_signal(self, normalized_query: str) -> bool:
        if re.search(r"\b(как купить|как оплатить|как продлить|способы оплаты|продление)\b", normalized_query):
            return False
        tokens = set(self._tokenize(normalized_query))
        if "цена" in tokens:
            return True
        if "подписк" in tokens:
            return True
        if "по" in normalized_query and "деньгам" in normalized_query:
            return True
        pricing_phrases = (
            "что по деньгам",
            "по деньгам",
            "сколько стоит",
            "какая стоимость",
            "какие тарифы",
        )
        return any(phrase in normalized_query for phrase in pricing_phrases)

    def _contains_multi_access_signal(self, normalized_query: str) -> bool:
        tokens = set(self._tokenize(normalized_query))
        stems = {
            "нескольк",
            "пользоват",
            "сотрудник",
            "люд",
            "дво",
            "многопользовательск",
            "устройств",
            "компьютер",
            "совместн",
            "рабоч",
            "мест",
        }
        return bool(tokens & stems)

    def _build_short_reply_boosts(
        self,
        normalized_query: str,
        context_signals: ContextSignals | None,
    ) -> dict[str, float]:
        if context_signals is None or context_signals.fallback_used or not self._is_short_reply_like(normalized_query):
            return {}
        context_tokens = set(self._tokenize(f"{context_signals.gist} {context_signals.meaning}"))
        boosts: dict[str, float] = {}
        if {"физлицо", "физик", "частн"} & context_tokens:
            boosts["physical_person_purchase"] = self._SHORT_REPLY_STATUS_BOOST
        if {"ип", "индивидуальн", "юрлиц", "автобизнес", "сто", "инн", "реквизит", "счет", "договор", "оформлен"} & context_tokens:
            boosts["legal_entity_purchase_flow"] = self._SHORT_REPLY_STATUS_BOOST
        if "демо" in context_tokens:
            boosts["demo_access"] = self._SHORT_REPLY_DEMO_BOOST
        return boosts

    def _is_tis_brand_context(self, normalized_query: str, context_signals: ContextSignals | None) -> bool:
        combined_context = normalized_query
        if context_signals is not None and not context_signals.fallback_used:
            combined_context = f"{combined_context} {self._normalize_text(context_signals.gist)} {self._normalize_text(context_signals.meaning)}"
        return bool(re.search(r"\b(tis|тис)\b", combined_context))

    def _is_brand_list_request(self, normalized_query: str, brand_hits: set[str]) -> bool:
        if brand_hits:
            return False
        return any(pattern.search(normalized_query) for pattern in self._BRAND_LIST_PATTERNS)

    def _is_catalog_list_request(self, normalized_query: str, brand_hits: set[str]) -> bool:
        if brand_hits:
            return False
        return any(pattern.search(normalized_query) for pattern in self._CATALOG_LIST_PATTERNS)

    def _is_partial_catalog_request(self, normalized_query: str, brand_hits: set[str]) -> bool:
        if brand_hits:
            return False
        return any(pattern.search(normalized_query) for pattern in self._PARTIAL_CATALOG_PATTERNS)

    def _is_short_reply_like(self, normalized_query: str) -> bool:
        tokens = self._tokenize(normalized_query)
        if not tokens:
            return False
        joined = " ".join(tokens)
        if joined in {"да", "нет", "ага", "неа", "юрлиц", "являюсь"}:
            return True
        return joined == "пока изуча" or (len(tokens) <= 3 and all(token in {"да", "нет", "ага", "неа", "юрлиц", "являюсь", "пока", "изуча"} for token in tokens))

    def _safe_fallback_shortlist(
        self,
        ranked: list[TopicShortlistCandidate],
        brand_hits: set[str],
        raw_query: str,
        limit: int,
    ) -> list[TopicShortlistCandidate]:
        selected_ids: list[str] = []
        if self._is_noise_like(raw_query) and "nonsense_input" in self._topics and "nonsense_input" not in selected_ids:
            selected_ids.append("nonsense_input")
        for topic_id in self._SAFE_FALLBACK_IDS:
            if topic_id in self._topics and topic_id not in selected_ids:
                selected_ids.append(topic_id)
            if len(selected_ids) >= limit:
                break

        candidate_map = {item.topic_id: item for item in ranked}
        return [candidate_map[topic_id] for topic_id in selected_ids if topic_id in candidate_map][:limit]

    def _score_source_tokens(self, source_tokens: set[str], signals: _TopicSignals) -> _SourceScore:
        if not source_tokens:
            return _SourceScore(0.0, 0.0, 0.0, 0.0, set())
        overlap = self._token_overlap(source_tokens, signals.positive_tokens)
        jaccard = self._jaccard(source_tokens, signals.positive_tokens)
        avoid_overlap = self._avoid_overlap(source_tokens, signals.anti_tokens)
        score = overlap + (0.55 * jaccard) - (0.35 * avoid_overlap)
        return _SourceScore(
            score=max(score, 0.0),
            overlap=overlap,
            jaccard=jaccard,
            avoid_overlap=avoid_overlap,
            matched_tokens=source_tokens & signals.positive_tokens,
        )

    def _build_topic_signals(self, topic: TopicDefinition) -> _TopicSignals:
        positive_parts = [
            topic.id,
            topic.title,
            " ".join(topic.keywords),
            " ".join(topic.aliases),
            " ".join(topic.examples),
            topic.description,
            topic.choose_when,
        ]
        anti_parts = [" ".join(topic.anti_keywords), topic.not_choose_when]
        return _TopicSignals(
            positive_tokens=set(self._tokenize(" ".join(part for part in positive_parts if part))),
            anti_tokens=set(self._tokenize(" ".join(part for part in anti_parts if part))),
        )

    def _build_query_tokens(self, query: str) -> set[str]:
        return set(self._tokenize(query))

    def _build_meaning_tokens(self, context_signals: ContextSignals | None) -> set[str]:
        if context_signals is None or context_signals.fallback_used:
            return set()
        return set(self._tokenize(context_signals.meaning))

    def _detect_brand_hits(self, query: str) -> tuple[set[str], set[str]]:
        normalized_query = self._normalize_text(query)
        query_tokens = set(self._tokenize(query))
        alias_hits: set[str] = set()
        fuzzy_hits: set[str] = set()
        for alias in self._brand_aliases:
            normalized_alias = self._normalize_text(alias)
            alias_tokens = set(self._tokenize(alias))
            alias_compact = normalized_alias.replace(" ", "")
            if normalized_alias and normalized_alias in normalized_query:
                alias_hits.add(normalized_alias)
                continue
            if alias_tokens and alias_tokens <= query_tokens:
                alias_hits.add(normalized_alias or alias)
                continue
            for token in query_tokens:
                if len(token) < 4:
                    continue
                similarity = SequenceMatcher(None, token, alias_compact).ratio()
                if similarity >= self._FUZZY_BRAND_THRESHOLD:
                    fuzzy_hits.add(normalized_alias or alias)
                    break
        return alias_hits, fuzzy_hits

    @classmethod
    def _load_brand_aliases(cls, path: Path) -> list[str]:
        if not path.exists():
            return []
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        aliases: set[str] = set()
        for item in raw.get("brands", []):
            if not isinstance(item, dict):
                continue
            canonical = str(item.get("canonical", "")).strip()
            if canonical:
                aliases.add(canonical)
            for alias in item.get("aliases", []):
                value = str(alias).strip()
                if value:
                    aliases.add(value)
        return sorted(aliases)

    @classmethod
    def _contains_abuse(cls, normalized_query: str) -> bool:
        return any(pattern.search(normalized_query) for pattern in cls._ABUSE_PATTERNS)

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        normalized = cls._normalize_text(text)
        tokens = [match.group(0) for match in cls._TOKEN_RE.finditer(normalized)]
        cleaned: list[str] = []
        for token in tokens:
            value = cls._normalize_token(token)
            if not value:
                continue
            if value in cls._STOPWORDS and value not in cls._KEEP_TOKENS:
                continue
            cleaned.append(value)
        return cleaned

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        value = str(token or "").strip().lower().replace("ё", "е")
        if not value:
            return ""
        value = cls._TOKEN_NORMALIZATIONS.get(value, value)
        for suffix in ("ами", "ями", "ого", "ему", "ому", "иях", "ах", "ях", "ов", "ев", "ей", "ам", "ям", "ом", "ем"):
            if len(value) > len(suffix) + 2 and value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        if len(value) > 4 and value.endswith(("а", "я", "ы", "и", "е", "у", "ю", "о")):
            value = value[:-1]
        return cls._TOKEN_NORMALIZATIONS.get(value, value)

    @staticmethod
    def _token_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(len(left), 1)

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _avoid_overlap(left: set[str], anti_tokens: set[str]) -> float:
        if not left or not anti_tokens:
            return 0.0
        return len(left & anti_tokens) / max(len(left), 1)

    @classmethod
    def _is_noise_like(cls, raw_query: str) -> bool:
        normalized = cls._normalize_text(raw_query)
        if not normalized:
            return True
        tokens = set(cls._tokenize(raw_query))
        if not tokens:
            return True
        if re.fullmatch(r"[.\W_0-9]+", str(raw_query).strip()) and len(tokens) <= 1:
            return True
        if any(char.isdigit() for char in raw_query) and any(len(token) >= 6 and len(set(token)) <= 3 for token in tokens):
            return True
        if len(tokens) == 1:
            token = next(iter(tokens))
            if len(token) >= 6 and len(set(token)) <= 2:
                return True
        return False
