from __future__ import annotations

import re

from the_First_Agent.context.prompt_context import PromptContext


class PromptContextExtractor:
    def extract(self, prompt: str) -> PromptContext:
        query_marker = self._query_marker(prompt)
        history_text = self._extract_block(
            prompt,
            "Диалог:\n",
            f"\n\n{query_marker}\n",
        )
        prompt_before_query, user_query, prompt_after_query = self._extract_user_query_parts(prompt)
        history_lines = [line.strip() for line in history_text.splitlines() if line.strip()]
        return PromptContext(
            history_text=history_text,
            user_query=user_query,
            history_lines=history_lines,
            last_assistant_message=self._last_assistant_line(history_lines),
            prompt_before_query=prompt_before_query,
            prompt_after_query=prompt_after_query,
        )

    @staticmethod
    def _extract_block(text: str, start_marker: str, end_marker: str | None) -> str:
        start_index = text.find(start_marker)
        if start_index < 0:
            return ""
        content_start = start_index + len(start_marker)
        if end_marker is None:
            return text[content_start:].strip()
        end_index = text.find(end_marker, content_start)
        if end_index < 0:
            return text[content_start:].strip()
        return text[content_start:end_index].strip()

    @staticmethod
    def _extract_user_query_parts(prompt: str) -> tuple[str, str, str]:
        end_match = re.search(r"\n\n(?:Краткий статус диалога|Контекст диалога|Формат ответа):\n", prompt)
        if not end_match:
            return "", "", ""
        query_end = end_match.start()
        marker = f"{PromptContextExtractor._query_marker(prompt)}\n"
        query_start = prompt.rfind(marker, 0, query_end)
        if query_start < 0:
            return "", "", ""
        content_start = query_start + len(marker)
        return prompt[:content_start], prompt[content_start:query_end].strip(), prompt[query_end:]

    @staticmethod
    def _query_marker(prompt: str) -> str:
        if "Последняя реплика клиента:\n" in prompt:
            return "Последняя реплика клиента:"
        if "Последняя фраза клиента:\n" in prompt:
            return "Последняя фраза клиента:"
        return "Перефразированная последняя фраза клиента:"

    @staticmethod
    def _last_assistant_line(history_lines: list[str]) -> str:
        fallback_markers = (
            "не удалось корректно сформировать ответ",
            "уточните запрос",
        )
        for line in reversed(history_lines):
            if line.lower().startswith(("assistant:", "бот:")):
                raw = line.split(":", 1)[1].strip()
                low = raw.lower()
                if any(marker in low for marker in fallback_markers):
                    continue
                return raw
        for line in reversed(history_lines):
            raw = line.split(":", 1)[1].strip() if ":" in line else line
            low = raw.lower()
            if any(marker in low for marker in fallback_markers):
                continue
            if "?" in raw:
                return raw
        if history_lines:
            raw = history_lines[-1]
            final = raw.split(":", 1)[1].strip() if ":" in raw else raw
            low = final.lower()
            if any(marker in low for marker in fallback_markers):
                return ""
            return final
        return ""
