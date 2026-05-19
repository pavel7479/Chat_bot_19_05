from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.agents.policy.types import ResponseAction


def _load_response_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


class ResponseAntiRepeatPolicy:
    def __init__(self, response_policy_file_path: Path) -> None:
        policy = _load_response_policy(response_policy_file_path)
        anti = policy.get("anti_repeat", {}) if isinstance(policy.get("anti_repeat", {}), dict) else {}
        variants_raw = anti.get("variants", {})
        self._action_variants: dict[str, list[str]] = {}
        if isinstance(variants_raw, dict):
            for action, variants in variants_raw.items():
                if not isinstance(variants, list):
                    continue
                items = [str(v).strip() for v in variants if str(v).strip()]
                if items:
                    self._action_variants[str(action).strip()] = items
        meta_pattern = str(
            anti.get(
                "meta_followup_pattern",
                r"^\s*(а\s+)?(подскажи|подробнее|короче|еще|ещ[её]|еще короче|ещ[её]\s+короче|повтори)\s*$",
            )
        )
        self._meta_followup_pattern = re.compile(meta_pattern)

    def apply(self, action: ResponseAction, answer_text: str, history_text: str, user_query: str) -> str:
        recent_answers = self._recent_assistant_answers(history_text, limit=2)
        if not recent_answers:
            return answer_text
        normalized_recent = {self._normalize(item) for item in recent_answers}
        if self._normalize(answer_text) not in normalized_recent:
            return answer_text
        for variant in self._action_variants.get(action.name, []):
            if self._normalize(variant) not in normalized_recent:
                return variant
        return answer_text

    @staticmethod
    def _recent_assistant_answers(history_text: str, limit: int = 2) -> list[str]:
        lines = [line.strip() for line in history_text.splitlines() if line.strip()]
        answers: list[str] = []
        current: list[str] = []
        collecting = False
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("user:"):
                if collecting and current:
                    answers.append("\n".join(reversed(current)).strip())
                    current = []
                    collecting = False
                continue
            if lowered.startswith("assistant:"):
                text = line.split(":", 1)[1].strip()
                if text:
                    current.append(text)
                answers.append("\n".join(reversed(current)).strip())
                current = []
                collecting = False
                if len(answers) >= limit:
                    break
                continue
            current.append(line)
            collecting = True
        return answers

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()
