from __future__ import annotations

import re

from src.core.models import BotResponse, SessionState


class GreetingService:
    _GREETING_ONLY_RE = re.compile(r"^\s*(привет|здравствуй|здравствуйте|добрый день|добрый вечер|hello|hi)\s*[!.?,-]*\s*$", re.IGNORECASE)
    _LEADING_GREETING_RE = re.compile(r"^\s*(здравствуйте|добрый день|добрый вечер|привет)[!,\.\s-]*", re.IGNORECASE)

    def should_short_circuit(
        self,
        user_query: str,
        session_state: SessionState,
        agent_zero_turn_type: str = "",
    ) -> bool:
        if session_state.greeted:
            return False
        if str(agent_zero_turn_type or "").strip() == "greeting":
            return True
        return bool(self._GREETING_ONLY_RE.match(str(user_query or "")))

    def build_greeting_response(self) -> str:
        return "Здравствуйте! Чем могу помочь?"

    def apply(self, answer_text: str, state_before_response: SessionState) -> str:
        text = str(answer_text or "").strip()
        if not text:
            text = "Чем могу помочь?"
        if state_before_response.greeted:
            stripped = self._LEADING_GREETING_RE.sub("", text).strip()
            return stripped or text
        if self._LEADING_GREETING_RE.match(text):
            return text
        return f"Здравствуйте! {text}".strip()

    def build_short_circuit_bot_response(self) -> BotResponse:
        return BotResponse(
            answer_text=self.build_greeting_response(),
            reasoning_summary="Детерминированное действие: greeting_once. Темы: ['nonsense_input']. evidence=knowledge_context; hits=1",
            topic_id="nonsense_input",
            topic_ids=["nonsense_input"],
            sources=["facts.yaml"],
            action_name="greeting_once",
            planned_action="greeting_once",
            used_evidence_ids=["greeting_once_service"],
            answer_sections=["greeting_once_service"],
            contract_flags={"planned_action_matches": True, "trace_complete": True},
        )
