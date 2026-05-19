from __future__ import annotations

import json
import re
from pathlib import Path


class PromptManager:
    _BACKTICK_ID_RE = re.compile(r"`([a-z_]+)`")
    _JSON_INTENT_ID_RE = re.compile(r'"intent_id"\s*:\s*"([a-z_]+)"')

    def __init__(self, project_root: Path, topic_prompt_path: str, answer_prompt_path: str) -> None:
        self._topic_prompt = (project_root / topic_prompt_path).read_text(encoding="utf-8")
        self._answer_prompt = (project_root / answer_prompt_path).read_text(encoding="utf-8")

    def build_topic_prompt(
        self,
        allowed_intents_text: str,
        topics_text: str,
        dynamic_rules_text: str,
        dynamic_examples_text: str,
        history_text: str,
        user_query: str,
        session_state_json: str = "{}",
        topic_title_map_json: str = "{}",
        context_understanding_text: str = "- Суть диалога: Не удалось надежно определить суть диалога.\n- Смысл последней реплики: Последняя реплика клиента не была дополнительно интерпретирована.",
    ) -> str:
        dialog_text = self._to_client_bot_dialog(history_text=history_text, user_query=user_query)
        session_state_summary = self._build_session_state_summary(session_state_json, topic_title_map_json)
        allowed_ids = self._extract_allowed_intent_ids(allowed_intents_text)
        safe_dynamic_rules_text = self._filter_dynamic_rules_text(dynamic_rules_text, allowed_ids)
        safe_dynamic_examples_text = self._filter_dynamic_examples_text(dynamic_examples_text, allowed_ids)
        return self._render(
            self._topic_prompt,
            {
                "allowed_intents_text": allowed_intents_text,
                "topics_text": topics_text,
                "dynamic_rules_text": safe_dynamic_rules_text,
                "dynamic_examples_text": safe_dynamic_examples_text,
                "history_text": history_text,
                "dialog_text": dialog_text,
                "user_query": user_query,
                "session_state_json": session_state_json,
                "session_state_summary": session_state_summary,
                "context_understanding_text": context_understanding_text,
            },
        )

    def build_answer_prompt(
        self,
        dialog_text: str,
        rewritten_query: str,
        intents_with_scores: str,
        normalized_brands: str,
        retrieved_facts_text: str,
        user_query: str,
    ) -> str:
        return self._render(
            self._answer_prompt,
            {
                "dialog_text": dialog_text,
                "rewritten_query": rewritten_query,
                "intents_with_scores": intents_with_scores,
                "normalized_brands": normalized_brands,
                "retrieved_facts_text": retrieved_facts_text,
                "user_query": user_query,
            },
        )

    def build_dialog_text(self, history_text: str, user_query: str) -> str:
        return self._to_client_bot_dialog(history_text=history_text, user_query=user_query)

    @staticmethod
    def build_context_understanding_text(gist: str, meaning: str) -> str:
        safe_gist = str(gist or "").strip() or "Не удалось надежно определить суть диалога."
        safe_meaning = str(meaning or "").strip() or "Последняя реплика клиента не была дополнительно интерпретирована."
        return "\n".join(
            [
                f"- Суть диалога: {safe_gist}",
                f"- Смысл последней реплики: {safe_meaning}",
            ]
        )

    @staticmethod
    def _render(template: str, values: dict[str, str]) -> str:
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return values.get(key, match.group(0))

        return re.sub(r"\{([a-z_]+)\}", replace, template)

    @staticmethod
    def _to_client_bot_dialog(history_text: str, user_query: str) -> str:
        lines: list[str] = []
        for raw in str(history_text).splitlines():
            row = raw.strip()
            if not row or row == "(история пуста)":
                continue
            if row.lower().startswith("user:"):
                lines.append(f"клиент: {row.split(':', 1)[1].strip()}")
            elif row.lower().startswith("assistant:"):
                lines.append(f"бот: {row.split(':', 1)[1].strip()}")
            elif row.lower().startswith("клиент:"):
                lines.append(f"клиент: {row.split(':', 1)[1].strip()}")
            elif row.lower().startswith("бот:"):
                lines.append(f"бот: {row.split(':', 1)[1].strip()}")
            else:
                lines.append(row)
        lines.append(f"клиент: {str(user_query).strip()}")
        return "\n".join(lines)

    @staticmethod
    def _build_session_state_summary(session_state_json: str, topic_title_map_json: str) -> str:
        try:
            state = json.loads(session_state_json or "{}")
        except Exception:
            state = {}
        if not isinstance(state, dict):
            state = {}
        try:
            topic_title_map = json.loads(topic_title_map_json or "{}")
        except Exception:
            topic_title_map = {}
        if not isinstance(topic_title_map, dict):
            topic_title_map = {}

        last_primary_topic = str(state.get("last_primary_topic", "out_of_scope_request")).strip() or "out_of_scope_request"
        last_bot_question_type = str(state.get("last_bot_question_type", "unknown")).strip() or "unknown"
        last_mentioned_brand = str(state.get("last_mentioned_brand", "")).strip() or "-"

        topic_title = lambda topic_id: str(topic_title_map.get(str(topic_id), str(topic_id or "-"))).strip() or "-"
        last_question_ru = {
            "unknown": "неизвестно",
            "brand_clarification": "уточнение бренда",
            "pricing": "уточнение цены",
            "demo_legal_check": "уточнение по демо",
            "legal_status_check": "уточнение статуса клиента",
            "invoice_confirmation": "подтверждение счета",
            "purchase_confirmation": "подтверждение оформления",
        }.get(last_bot_question_type, last_bot_question_type)

        return "\n".join(
            [
                f"Последняя тема: {topic_title(last_primary_topic)}",
                f"Последний вопрос бота: {last_question_ru}",
                f"Последний бренд: {last_mentioned_brand}",
            ]
        )

    @classmethod
    def _extract_allowed_intent_ids(cls, allowed_intents_text: str) -> set[str]:
        allowed: set[str] = set()
        for raw_line in str(allowed_intents_text or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            value = line[2:].strip()
            if value:
                allowed.add(value)
        return allowed

    @classmethod
    def _filter_dynamic_rules_text(cls, dynamic_rules_text: str, allowed_ids: set[str]) -> str:
        sections: list[str] = []
        for raw_section in str(dynamic_rules_text or "").split("\n\n"):
            section = raw_section.strip()
            if not section:
                continue
            lines = section.splitlines()
            header = lines[0]
            body = lines[1:]
            filtered_body = [line for line in body if cls._chunk_is_compatible(line, allowed_ids)]
            if filtered_body:
                sections.append("\n".join([header, *filtered_body]))
        return "\n\n".join(sections)

    @classmethod
    def _filter_dynamic_examples_text(cls, dynamic_examples_text: str, allowed_ids: set[str]) -> str:
        safe_chunks: list[str] = []
        for raw_chunk in cls._split_example_blocks(dynamic_examples_text):
            chunk = raw_chunk.strip()
            if not chunk:
                continue
            if cls._chunk_is_compatible(chunk, allowed_ids):
                safe_chunks.append(chunk)
        return "\n\n".join(safe_chunks)

    @staticmethod
    def _split_example_blocks(dynamic_examples_text: str) -> list[str]:
        text = str(dynamic_examples_text or "").strip()
        if not text:
            return []

        blocks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.strip() == "Пример:" and current:
                blocks.append("\n".join(current).strip())
                current = [line]
                continue
            current.append(line)
        if current:
            blocks.append("\n".join(current).strip())
        return blocks

    @classmethod
    def _chunk_is_compatible(cls, text: str, allowed_ids: set[str]) -> bool:
        referenced = cls._extract_referenced_intent_ids(text)
        return referenced.issubset(allowed_ids)

    @classmethod
    def _extract_referenced_intent_ids(cls, text: str) -> set[str]:
        referenced: set[str] = set()
        for pattern in (cls._BACKTICK_ID_RE, cls._JSON_INTENT_ID_RE):
            referenced.update(pattern.findall(str(text or "")))
        return referenced
