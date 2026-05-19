from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    name: str
    session_id: str


@dataclass(slots=True)
class LLMConfig:
    provider: str
    model_name: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout: int


@dataclass(slots=True)
class PathsConfig:
    knowledge_base_path: str
    topics_file: str
    topic_classifier_prompt: str
    context_understanding_prompt: str
    answer_generator_prompt: str
    intents_config_file: str
    brands_file: str
    response_policy_file: str
    prices_file: str
    intent_model_file: str = "src/config/intent_model.yaml"


@dataclass(slots=True)
class RetrievalConfig:
    type: str
    top_k: int
    max_context_chars: int
    bm25_weight: float = 0.55
    dense_weight: float = 0.45
    min_evidence_score: float = 0.15
    min_evidence_hits: int = 1


@dataclass(slots=True)
class SessionConfig:
    max_history_messages: int


@dataclass(slots=True)
class LoggingConfig:
    level: str
    file_path: str
    console_enabled: bool
    json_format: bool


@dataclass(slots=True)
class ApiConfig:
    host: str
    port: int


@dataclass(slots=True)
class Config:
    app: AppConfig
    llm: LLMConfig
    paths: PathsConfig
    retrieval: RetrievalConfig
    session: SessionConfig
    logging: LoggingConfig
    api: ApiConfig
