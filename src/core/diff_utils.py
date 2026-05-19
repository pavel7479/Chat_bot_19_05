from __future__ import annotations

from copy import deepcopy


def snapshot(value: object) -> object:
    return deepcopy(value)


def dict_diff(before: object, after: object, path: str = "") -> list[dict[str, object]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, object]] = []
        keys = sorted(set(before.keys()) | set(after.keys()), key=str)
        for key in keys:
            next_path = f"{path}.{key}" if path else str(key)
            before_value = before.get(key)
            after_value = after.get(key)
            changes.extend(dict_diff(before_value, after_value, next_path))
        return changes

    if before != after:
        return [
            {
                "path": path or "$",
                "before": deepcopy(before),
                "after": deepcopy(after),
            }
        ]
    return []
