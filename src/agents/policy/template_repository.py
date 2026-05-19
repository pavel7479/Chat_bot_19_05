from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ResponseTemplateRepository:
    """Read-only repository for response templates from policy config."""

    def __init__(self, response_policy_file_path: Path) -> None:
        self._path = response_policy_file_path
        self._templates = self._load_templates()

    def get(self, action_name: str) -> list[str]:
        return list(self._templates.get(action_name, []))

    def has(self, action_name: str) -> bool:
        return action_name in self._templates and bool(self._templates[action_name])

    def _load_templates(self) -> dict[str, list[str]]:
        if not self._path.exists():
            return {}
        raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            return {}
        templates_raw = raw.get("templates", {})
        if not isinstance(templates_raw, dict):
            return {}
        templates: dict[str, list[str]] = {}
        for action, values in templates_raw.items():
            if not isinstance(values, list):
                continue
            items = [str(item).strip() for item in values if str(item).strip()]
            if items:
                templates[str(action).strip()] = items
        return templates

