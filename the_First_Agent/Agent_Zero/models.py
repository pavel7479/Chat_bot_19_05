from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.core.models import SemanticFrame


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SemanticFrameSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_mode: str = Field(default="unknown")
    user_goal: str = Field(default="unknown")
    is_followup: bool = Field(default=False)
    is_topic_switch: bool = Field(default=False)
    language: str = Field(default="ru")
    confidence: float = Field(default=0.0)
    gist: str = Field(default="")
    meaning: str = Field(default="")


class ContextUnderstandingSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gist: NonEmptyText = Field()
    meaning: NonEmptyText = Field()
    turn_type: str = Field(default="other")
    turn_subtype: str = Field(default="")
    confidence: float = Field(default=0.0)
    semantic_flags: list[str] = Field(default_factory=list)
    semantic_frame: SemanticFrameSchema | None = Field(default=None)


@dataclass(slots=True)
class ContextUnderstandingResult:
    gist: str
    meaning: str
    raw_response: str
    parsed_json: dict[str, object]
    turn_type: str = "other"
    turn_subtype: str = ""
    confidence: float = 0.0
    semantic_flags: list[str] = None
    semantic_frame: SemanticFrame | None = None
    fallback_used: bool = False
    fallback_reason: str = ""
    json_extracted_from_wrapped_response: bool = False
    schema_retry_used: bool = False
    validation_error: str = ""

    def __post_init__(self) -> None:
        if self.semantic_flags is None:
            self.semantic_flags = []
        if self.semantic_frame is None:
            self.semantic_frame = build_semantic_frame(
                user_query="",
                gist=self.gist,
                meaning=self.meaning,
                turn_type=self.turn_type,
                confidence=self.confidence,
            )


def build_semantic_frame(
    *,
    user_query: str,
    gist: str,
    meaning: str,
    turn_type: str,
    confidence: float,
    provided_frame: SemanticFrameSchema | None = None,
) -> SemanticFrame:
    if provided_frame is None:
        return _build_neutral_semantic_frame(
            gist=gist,
            meaning=meaning,
            confidence=confidence,
        )
    return SemanticFrame(
        conversation_mode=str(provided_frame.conversation_mode or "unknown").strip() or "unknown",
        user_goal=str(provided_frame.user_goal or "unknown").strip() or "unknown",
        is_followup=bool(provided_frame.is_followup),
        is_topic_switch=bool(provided_frame.is_topic_switch),
        language=str(provided_frame.language or "ru").strip() or "ru",
        confidence=float(provided_frame.confidence or confidence or 0.0),
        gist=str(provided_frame.gist or gist).strip() or gist,
        meaning=str(provided_frame.meaning or meaning).strip() or meaning,
    )


def _build_neutral_semantic_frame(
    *,
    gist: str,
    meaning: str,
    confidence: float,
) -> SemanticFrame:
    return SemanticFrame(
        conversation_mode="unknown",
        user_goal="unknown",
        is_followup=False,
        is_topic_switch=False,
        language="ru",
        confidence=float(confidence or 0.0),
        gist=str(gist or "").strip(),
        meaning=str(meaning or "").strip(),
    )
