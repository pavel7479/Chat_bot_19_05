from __future__ import annotations

from collections import defaultdict

from src.core.models import ChatMessage, SessionState


class SessionManager:
    def __init__(self, max_history_messages: int) -> None:
        self._max_history_messages = max_history_messages
        self._messages: dict[str, list[ChatMessage]] = defaultdict(list)
        self._state: dict[str, SessionState] = defaultdict(SessionState)

    def add_user_message(self, session_id: str, text: str) -> None:
        self._messages[session_id].append(ChatMessage(role="user", text=text))
        self._trim(session_id)

    def add_bot_message(self, session_id: str, text: str) -> None:
        self._messages[session_id].append(ChatMessage(role="assistant", text=text))
        self._trim(session_id)

    def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self._messages[session_id])

    def get_state(self, session_id: str) -> SessionState:
        state = self._state[session_id]
        return SessionState(**state.as_dict())

    def set_state(self, session_id: str, state: SessionState) -> None:
        self._state[session_id] = SessionState(**state.as_dict())

    def clear(self, session_id: str) -> None:
        self._messages[session_id].clear()
        self._state.pop(session_id, None)

    def _trim(self, session_id: str) -> None:
        if len(self._messages[session_id]) > self._max_history_messages:
            self._messages[session_id] = self._messages[session_id][-self._max_history_messages :]
