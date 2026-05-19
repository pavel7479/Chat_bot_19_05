from __future__ import annotations

from pydantic import BaseModel, Field


class IntentChoice(BaseModel):
    intent_id: str = Field(min_length=1, description="Системный идентификатор темы строго из списка тем в prompt")
    score: float = Field(ge=0.0, le=1.0, description="Оценка релевантности темы от 0 до 1")
    reason: str = Field(min_length=1, description="Короткое объяснение, почему выбрана эта тема")


class FirstAgentOutputSchema(BaseModel):
    intent_1: IntentChoice = Field(description="Главная тема для ответа на последнюю фразу клиента")
    intent_2: IntentChoice | None = Field(
        default=None,
        description="Вторая тема, если в последней фразе клиента действительно есть второй смысл",
    )
