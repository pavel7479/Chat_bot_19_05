from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.core.models import ClassifierState, StateDiff


class StateTraceService:
    """Applies state patches and records a normalized step diff."""

    @staticmethod
    def apply_patch(
        state: ClassifierState,
        *,
        step: str,
        patch: dict[str, Any],
        reason: str,
    ) -> None:
        before = {
            "resolved_intents": list(state.resolved_intents),
            "signals": list(state.signals),
            "current_focus": state.current_focus,
            "planned_action": state.planned_action,
            "flow_name": state.flow_name,
            "flow_step": state.flow_step,
            "confidence": state.confidence,
        }

        for key, value in patch.items():
            if hasattr(state, key):
                setattr(state, key, value)

        after = {
            "resolved_intents": list(state.resolved_intents),
            "signals": list(state.signals),
            "current_focus": state.current_focus,
            "planned_action": state.planned_action,
            "flow_name": state.flow_name,
            "flow_step": state.flow_step,
            "confidence": state.confidence,
        }

        state.trace.append(
            StateDiff(
                step=step,
                before=before,
                patch={k: StateTraceService._safe(v) for k, v in patch.items()},
                after=after,
                reason=reason,
            )
        )

    @staticmethod
    def _safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): StateTraceService._safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [StateTraceService._safe(v) for v in value]
        try:
            return asdict(value)
        except Exception:
            return str(value)
