from __future__ import annotations

import logging
from collections import Counter


class TestResultLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._failed_cases: list[str] = []
        self._failure_reasons: Counter[str] = Counter()

    def record_case(self, case_name: str, status: str, fallback_reason: str) -> None:
        if status != "FAIL":
            return
        self._failed_cases.append(case_name)
        reason = str(fallback_reason).strip() or "none"
        self._failure_reasons[reason] += 1

    def emit_summary(self) -> None:
        self._logger.info("failed_cases_list | count=%s | cases=%s", len(self._failed_cases), self._failed_cases)
        self._logger.info("failed_cases_by_reason | reasons=%s", dict(self._failure_reasons))

