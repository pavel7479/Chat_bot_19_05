from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    client_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str = "ok"
