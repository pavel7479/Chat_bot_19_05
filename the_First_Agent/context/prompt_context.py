from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PromptContext:
    history_text: str
    user_query: str
    history_lines: list[str]
    last_assistant_message: str
    prompt_before_query: str = ""
    prompt_after_query: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.history_text}\n{self.user_query}".lower()

    @property
    def history_lower(self) -> str:
        return self.history_text.lower()

    @property
    def query_lower(self) -> str:
        return self.user_query.lower()
