from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


@dataclass(slots=True)
class ChatMessage:
    role: str
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class TopicDefinition:
    id: str
    title: str
    description: str
    keywords: list[str]
    aliases: list[str] = field(default_factory=list)
    anti_keywords: list[str] = field(default_factory=list)
    choose_when: str = ""
    not_choose_when: str = ""
    choose_when_points: list[str] = field(default_factory=list)
    not_choose_when_points: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ContextSignals:
    user_query: str
    gist: str
    meaning: str
    semantic_flags: set[str] = field(default_factory=set)
    semantic_boost_topics: set[str] = field(default_factory=set)
    semantic_penalty_topics: set[str] = field(default_factory=set)
    continuity_topics: set[str] = field(default_factory=set)
    fallback_used: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "user_query": self.user_query,
            "gist": self.gist,
            "meaning": self.meaning,
            "semantic_flags": sorted(self.semantic_flags),
            "semantic_boost_topics": sorted(self.semantic_boost_topics),
            "semantic_penalty_topics": sorted(self.semantic_penalty_topics),
            "continuity_topics": sorted(self.continuity_topics),
            "fallback_used": self.fallback_used,
        }


@dataclass(slots=True)
class SemanticFrame:
    conversation_mode: str = "unknown"
    user_goal: str = "unknown"
    is_followup: bool = False
    is_topic_switch: bool = False
    language: str = "ru"
    confidence: float = 0.0
    gist: str = ""
    meaning: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "conversation_mode": self.conversation_mode,
            "user_goal": self.user_goal,
            "is_followup": self.is_followup,
            "is_topic_switch": self.is_topic_switch,
            "language": self.language,
            "confidence": self.confidence,
            "gist": self.gist,
            "meaning": self.meaning,
        }


@dataclass(slots=True)
class TopicClassificationResult:
    topic_ids: list[str]
    confidence: float
    reason: str
    current_focus: str = "unknown"
    planned_action: str = ""
    secondary_actions: list[str] = field(default_factory=list)
    response_plan: list[str] = field(default_factory=list)
    clarify_required: bool = False
    nonsense_input: bool = False
    abuse_input: bool = False
    flow_name: str = "none"
    flow_step: str = "idle"
    retrieval_context: dict[str, object] = field(default_factory=dict)
    rule_trace: list[dict[str, object]] = field(default_factory=list)
    state_snapshot: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    classifier_source: str = "legacy"
    fallback_reason: str = ""

    @property
    def primary_topic_id(self) -> str:
        return self.topic_ids[0] if self.topic_ids else "out_of_scope_request"


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    score: float
    source: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FactRecord:
    fact_id: str
    topic: str
    subtopic: str
    entity_type: str
    entity: str
    text: str
    fact_type: str = "knowledge"
    section_tag: str = "general"
    priority: int = 0
    aliases: list[str] = field(default_factory=list)
    action_tags: list[str] = field(default_factory=list)
    template: str = ""
    required_slots: list[str] = field(default_factory=list)
    semantic_group: str = ""


@dataclass(slots=True)
class MediaReference:
    media_type: str
    url: str
    description: str
    trigger_condition: str = ""


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    text: str
    score: float
    source: str
    action_name: str
    why_selected: str = ""
    section_tag: str = ""
    source_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalQueryContext:
    trace_id: str
    raw_query: str
    topic_ids: list[str] = field(default_factory=list)
    planned_action: str = ""
    current_focus: str = "unknown"
    slots_snapshot: dict[str, object] = field(default_factory=dict)
    state_snapshot: dict[str, object] = field(default_factory=dict)
    query_variants: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IntentPromptContext:
    trace_id: str
    user_query: str
    rewritten_query: str
    context_json: dict[str, object] = field(default_factory=dict)
    selected_intents: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TopicSelectionTrace:
    selected_intents: list[str] = field(default_factory=list)
    selected_reasons: dict[str, str] = field(default_factory=dict)
    dropped_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RewriteTrace:
    original_query: str
    rewritten_query: str
    changed: bool
    reason: str


@dataclass(slots=True)
class ModelParseTrace:
    parse_status: str = ""
    validation_error: str = ""
    retry_used: bool = False
    fallback_reason: str = ""


@dataclass(slots=True)
class DroppedTopicReason:
    topic_id: str
    dropped_by: str
    reason: str = ""


@dataclass(slots=True)
class EvidencePack:
    items: list[EvidenceItem] = field(default_factory=list)
    status: str = "unknown"


@dataclass(slots=True)
class MandatoryMeaningBlock:
    key: str
    required_phrases: list[str] = field(default_factory=list)
    semantic_tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "required_phrases": list(self.required_phrases),
            "semantic_tags": list(self.semantic_tags),
        }


@dataclass(slots=True)
class PreparedResponseContext:
    primary_facts: list[str] = field(default_factory=list)
    secondary_facts: list[str] = field(default_factory=list)
    prices: list[str] = field(default_factory=list)
    followup_questions: list[str] = field(default_factory=list)
    slots: dict[str, object] = field(default_factory=dict)
    product_context: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "primary_facts": list(self.primary_facts),
            "secondary_facts": list(self.secondary_facts),
            "prices": list(self.prices),
            "followup_questions": list(self.followup_questions),
            "slots": dict(self.slots),
            "product_context": dict(self.product_context),
        }


@dataclass(slots=True)
class BrandMention:
    raw_text: str
    normalized_key: str
    recognized: bool
    display_name: str
    canonical_brand: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "normalized_key": self.normalized_key,
            "recognized": self.recognized,
            "display_name": self.display_name,
            "canonical_brand": self.canonical_brand,
        }


class PricingBrandStatus(str, Enum):
    PRICED = "priced"
    MISSING_PRICE = "missing_price"
    UNRESOLVED = "unresolved"


class PricingFlowMode(str, Enum):
    ALL = "all"
    SEPARATE = "separate_processing"
    REMAINING_ONLY = "remaining_only"
    EXPLAIN_UNRESOLVED = "explain_unresolved"


@dataclass(slots=True)
class PricingBrandItem:
    canonical_brand: str
    display_name: str
    raw_surface: str
    recognized: bool
    has_price: bool = False
    processed: bool = False
    status: PricingBrandStatus = PricingBrandStatus.UNRESOLVED

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_brand": self.canonical_brand,
            "display_name": self.display_name,
            "raw_surface": self.raw_surface,
            "recognized": self.recognized,
            "has_price": self.has_price,
            "processed": self.processed,
            "status": self.status.value,
        }


@dataclass(slots=True)
class ServiceSemanticMemory:
    used_groups: list[str] = field(default_factory=list)
    used_fact_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "used_groups": list(self.used_groups),
            "used_fact_ids": list(self.used_fact_ids),
        }


@dataclass(slots=True)
class PricingFlowState:
    active: bool = False
    product: str = ""
    mode: PricingFlowMode = PricingFlowMode.ALL
    stage: str = "idle"
    brand_mentions: list[BrandMention] = field(default_factory=list)
    brand_items: list[PricingBrandItem] = field(default_factory=list)
    raw_brand_mentions: list[str] = field(default_factory=list)
    requested_brand_keys: list[str] = field(default_factory=list)
    recognized_brands: list[str] = field(default_factory=list)
    unknown_brand_mentions: list[str] = field(default_factory=list)
    priced_brands: list[str] = field(default_factory=list)
    missing_price_brands: list[str] = field(default_factory=list)
    pending_brand_mentions: list[str] = field(default_factory=list)
    processed_brand_mentions: list[str] = field(default_factory=list)
    remaining_brand_mentions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "product": self.product,
            "mode": self.mode.value if isinstance(self.mode, PricingFlowMode) else str(self.mode),
            "stage": self.stage,
            "brand_mentions": [item.as_dict() for item in self.brand_mentions],
            "brand_items": [item.as_dict() for item in self.brand_items],
            "raw_brand_mentions": list(self.raw_brand_mentions),
            "requested_brand_keys": list(self.requested_brand_keys),
            "recognized_brands": list(self.recognized_brands),
            "unknown_brand_mentions": list(self.unknown_brand_mentions),
            "priced_brands": list(self.priced_brands),
            "missing_price_brands": list(self.missing_price_brands),
            "pending_brand_mentions": list(self.pending_brand_mentions),
            "processed_brand_mentions": list(self.processed_brand_mentions),
            "remaining_brand_mentions": list(self.remaining_brand_mentions),
        }


@dataclass(slots=True)
class BotResponse:
    answer_text: str
    reasoning_summary: str
    topic_id: str
    topic_ids: list[str]
    sources: list[str]
    action_name: str = ""
    planned_action: str = ""
    used_evidence_ids: list[str] = field(default_factory=list)
    answer_sections: list[str] = field(default_factory=list)
    contract_flags: dict[str, bool] = field(default_factory=dict)
    media_refs: list[MediaReference] = field(default_factory=list)
    evidence_pack: EvidencePack = field(default_factory=EvidencePack)


@dataclass(slots=True)
class SessionState:
    client_type: str = "unknown"
    legal_entity_confirmed: bool = False
    purchase_active: bool = False
    purchase_stage: str = "unknown"
    last_question_type: str = "unknown"
    last_primary_topic: str = "out_of_scope_request"
    last_topic_ids: list[str] = field(default_factory=list)
    last_secondary_topics: list[str] = field(default_factory=list)
    dialog_phase: str = "discovery"
    conversation_closed: bool = False
    greeted: bool = False
    active_flow: str = "none"
    flow_step: str = "idle"
    last_action_name: str = ""
    same_action_repeats: int = 0
    manager_handoff_stage: str = "none"
    evidence_status: str = "unknown"
    active_request_kind: str = "none"
    document_contact_collected: bool = False
    last_bot_question_type: str = "unknown"
    last_mentioned_brand: str = ""
    last_focus_topic: str = "out_of_scope_request"
    last_context_gist: str = ""
    last_context_meaning: str = ""
    last_context_fallback_used: bool = False
    last_context_fallback_reason: str = ""
    conversation_mode: str = "unknown"
    user_goal: str = "unknown"
    last_semantic_frame: dict[str, object] = field(default_factory=dict)
    active_business_flow: str = "none"
    active_pricing_flow: str = "none"
    pricing_requested_product: str = ""
    requested_brands: list[str] = field(default_factory=list)
    recognized_brands: list[str] = field(default_factory=list)
    unknown_brand_mentions: list[str] = field(default_factory=list)
    missing_price_brands: list[str] = field(default_factory=list)
    priced_brands: list[str] = field(default_factory=list)
    pending_brand_mentions: list[str] = field(default_factory=list)
    pricing_mode: str = "all"
    pricing_flow: dict[str, object] = field(default_factory=dict)
    service_semantic_memory: dict[str, object] = field(default_factory=dict)
    slots: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "client_type": self.client_type,
            "legal_entity_confirmed": self.legal_entity_confirmed,
            "purchase_active": self.purchase_active,
            "purchase_stage": self.purchase_stage,
            "last_question_type": self.last_question_type,
            "last_primary_topic": self.last_primary_topic,
            "last_topic_ids": list(self.last_topic_ids),
            "last_secondary_topics": list(self.last_secondary_topics),
            "dialog_phase": self.dialog_phase,
            "conversation_closed": self.conversation_closed,
            "greeted": self.greeted,
            "active_flow": self.active_flow,
            "flow_step": self.flow_step,
            "last_action_name": self.last_action_name,
            "same_action_repeats": self.same_action_repeats,
            "manager_handoff_stage": self.manager_handoff_stage,
            "evidence_status": self.evidence_status,
            "active_request_kind": self.active_request_kind,
            "document_contact_collected": self.document_contact_collected,
            "last_bot_question_type": self.last_bot_question_type,
            "last_mentioned_brand": self.last_mentioned_brand,
            "last_focus_topic": self.last_focus_topic,
            "last_context_gist": self.last_context_gist,
            "last_context_meaning": self.last_context_meaning,
            "last_context_fallback_used": self.last_context_fallback_used,
            "last_context_fallback_reason": self.last_context_fallback_reason,
            "conversation_mode": self.conversation_mode,
            "user_goal": self.user_goal,
            "last_semantic_frame": dict(self.last_semantic_frame),
            "active_business_flow": self.active_business_flow,
            "active_pricing_flow": self.active_pricing_flow,
            "pricing_requested_product": self.pricing_requested_product,
            "requested_brands": list(self.requested_brands),
            "recognized_brands": list(self.recognized_brands),
            "unknown_brand_mentions": list(self.unknown_brand_mentions),
            "missing_price_brands": list(self.missing_price_brands),
            "priced_brands": list(self.priced_brands),
            "pending_brand_mentions": list(self.pending_brand_mentions),
            "pricing_mode": self.pricing_mode,
            "pricing_flow": dict(self.pricing_flow),
            "service_semantic_memory": dict(self.service_semantic_memory),
            "slots": dict(self.slots),
        }


@dataclass(slots=True)
class IntentCandidate:
    intent: str
    score: float
    reasons: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
