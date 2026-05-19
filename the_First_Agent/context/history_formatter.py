from __future__ import annotations

from src.core.models import ChatMessage


def format_history(messages: list[ChatMessage]) -> str:
    if not messages:
        return "(история пуста)"
    return "\n".join(f"{msg.role}: {msg.text}" for msg in messages)
