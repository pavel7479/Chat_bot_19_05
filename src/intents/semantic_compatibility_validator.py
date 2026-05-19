from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.models import SemanticFrame, SessionState


@dataclass(slots=True)
class SemanticCompatibilityResult:
    is_compatible: bool
    severity: str = "NONE"
    selected_topic: str = ""
    final_topic: str = ""
    allowed_topics: list[str] = field(default_factory=list)
    compatible_shortlist_topic: str = ""
    override_applied: bool = False
    fallback_topic_used: bool = False
    fallback_topic: str = ""
    reason: str = ""
    skipped: bool = False
    skip_reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "is_compatible": self.is_compatible,
            "severity": self.severity,
            "selected_topic": self.selected_topic,
            "final_topic": self.final_topic,
            "allowed_topics": list(self.allowed_topics),
            "compatible_shortlist_topic": self.compatible_shortlist_topic,
            "override_applied": self.override_applied,
            "fallback_topic_used": self.fallback_topic_used,
            "fallback_topic": self.fallback_topic,
            "reason": self.reason,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


class SemanticCompatibilityValidator:
    _HIGH_PRIORITY_MODES = {"support", "manager", "purchase", "security", "complaint"}
    _SAFE_FALLBACK_TOPIC = "nonsense_input"

    def __init__(self, mapping_path: Path) -> None:
        self._mapping = self._load_mapping(mapping_path)

    def validate(
        self,
        *,
        semantic_frame: SemanticFrame | None,
        topic_ids: list[str],
        shortlist_ids: list[str],
        state: SessionState,
    ) -> SemanticCompatibilityResult:
        del state
        selected_topic = str(topic_ids[0]).strip() if topic_ids else ""
        if semantic_frame is None:
            return SemanticCompatibilityResult(
                is_compatible=True,
                selected_topic=selected_topic,
                final_topic=selected_topic,
                skipped=True,
                skip_reason="semantic_frame_missing",
                reason="SemanticFrame missing; compatibility validation skipped.",
            )

        goal = str(semantic_frame.user_goal or "unknown").strip() or "unknown"
        mode = str(semantic_frame.conversation_mode or "unknown").strip() or "unknown"
        allowed_topics = list(self._mapping.get("user_goal_to_allowed_topics", {}).get(goal, []))
        if goal == "unknown" or mode == "unknown":
            return SemanticCompatibilityResult(
                is_compatible=True,
                severity="WARNING",
                selected_topic=selected_topic,
                final_topic=selected_topic,
                allowed_topics=allowed_topics,
                skipped=True,
                skip_reason="neutral_semantic_frame",
                reason="SemanticFrame is neutral; compatibility validation is observational only.",
            )

        if not allowed_topics:
            return SemanticCompatibilityResult(
                is_compatible=True,
                severity="WARNING",
                selected_topic=selected_topic,
                final_topic=selected_topic,
                allowed_topics=[],
                skipped=True,
                skip_reason="goal_not_mapped",
                reason=f"No semantic topic map configured for user_goal={goal}.",
            )

        if selected_topic in allowed_topics:
            return SemanticCompatibilityResult(
                is_compatible=True,
                selected_topic=selected_topic,
                final_topic=selected_topic,
                allowed_topics=allowed_topics,
                reason="Selected topic is compatible with SemanticFrame.",
            )

        compatible_shortlist_topic = next((topic_id for topic_id in shortlist_ids if topic_id in allowed_topics), "")
        severity = "HARD_CONFLICT" if mode in self._HIGH_PRIORITY_MODES else "SOFT_CONFLICT"
        if compatible_shortlist_topic:
            return SemanticCompatibilityResult(
                is_compatible=False,
                severity=severity,
                selected_topic=selected_topic,
                final_topic=compatible_shortlist_topic,
                allowed_topics=allowed_topics,
                compatible_shortlist_topic=compatible_shortlist_topic,
                override_applied=True,
                reason=(
                    f"Classifier selected incompatible topic `{selected_topic}` for user_goal={goal}; "
                    f"replaced with compatible shortlist topic `{compatible_shortlist_topic}`."
                ),
            )

        return SemanticCompatibilityResult(
            is_compatible=False,
            severity=severity,
            selected_topic=selected_topic,
            final_topic=self._SAFE_FALLBACK_TOPIC,
            allowed_topics=allowed_topics,
            fallback_topic_used=True,
            fallback_topic=self._SAFE_FALLBACK_TOPIC,
            reason=(
                f"Classifier selected incompatible topic `{selected_topic}` for user_goal={goal}; "
                f"no compatible shortlist topic found, fallback to `{self._SAFE_FALLBACK_TOPIC}`."
            ),
        )

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}
