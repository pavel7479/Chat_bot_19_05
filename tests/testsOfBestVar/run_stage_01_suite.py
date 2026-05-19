from __future__ import annotations

import sys
import time
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


STAGE_01_MODULES = [
    "tests.test_context_understanding_agent_01",
    "tests.test_context_understanding_parser_01",
    "tests.test_intent_agent_context_understanding_01",
    "tests.test_context_understanding_prompt_01",
    "tests.testsOfBestVar.test_stage_01_semantic_frame_light",
]


class Stage01LoggingResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity, *, log_path: Path) -> None:
        super().__init__(stream, descriptions, verbosity)
        self._log_path = log_path
        self._test_started_at: dict[str, float] = {}
        self._blocks: list[str] = []
        self._subtest_failures: list[tuple[str, tuple[object, object, object]]] = []
        self._subtest_errors: list[tuple[str, tuple[object, object, object]]] = []

    def startTest(self, test) -> None:
        test_id = test.id()
        self._test_started_at[test_id] = time.monotonic()
        super().startTest(test)

    def addSuccess(self, test) -> None:
        super().addSuccess(test)
        self._append_block(test=test, status="PASS")

    def addFailure(self, test, err) -> None:
        super().addFailure(test, err)
        self._append_block(test=test, status="FAIL", err=err)

    def addError(self, test, err) -> None:
        super().addError(test, err)
        self._append_block(test=test, status="ERROR", err=err)

    def addSubTest(self, test, subtest, err) -> None:
        super().addSubTest(test, subtest, err)
        if err is None:
            return
        status = "FAIL" if issubclass(err[0], test.failureException) else "ERROR"
        subtest_id = self._format_subtest_id(test, subtest)
        if status == "FAIL":
            self._subtest_failures.append((subtest_id, err))
        else:
            self._subtest_errors.append((subtest_id, err))
        self._append_block(test=test, status=status, err=err, test_id_override=subtest_id)

    def stopTestRun(self) -> None:
        super().stopTestRun()
        total_failures = len(self.failures) + len(self._subtest_failures)
        total_errors = len(self.errors) + len(self._subtest_errors)
        summary = [
            "=== SUMMARY ===",
            f"generated_at={datetime.now(timezone.utc).isoformat()}",
            f"tests_run={self.testsRun}",
            f"failures={total_failures}",
            f"errors={total_errors}",
            f"successful={self.testsRun - total_failures - total_errors}",
            "failed_tests="
            + ", ".join(
                [test.id() for test, _ in self.failures]
                + [subtest_id for subtest_id, _ in self._subtest_failures]
            )
            if (self.failures or self._subtest_failures)
            else "failed_tests=",
            "error_tests="
            + ", ".join(
                [test.id() for test, _ in self.errors]
                + [subtest_id for subtest_id, _ in self._subtest_errors]
            )
            if (self.errors or self._subtest_errors)
            else "error_tests=",
            "",
        ]
        self._log_path.write_text(
            "\n".join(
                [
                    "Stage 01 Test Run Log",
                    f"generated_at={datetime.now(timezone.utc).isoformat()}",
                    "modules=" + ", ".join(STAGE_01_MODULES),
                    "",
                    *summary,
                    *self._blocks,
                ]
            ),
            encoding="utf-8",
        )

    def _append_block(self, *, test, status: str, err=None, test_id_override: str | None = None) -> None:
        test_id = test_id_override or test.id()
        started_at = self._test_started_at.get(test_id)
        if started_at is None:
            started_at = self._test_started_at.get(test.id())
        duration_ms = 0.0 if started_at is None else round((time.monotonic() - started_at) * 1000, 2)
        block = [
            "=== TEST RESULT ===",
            f"test_id={test_id}",
            f"status={status}",
            f"duration_ms={duration_ms}",
        ]
        if err is not None:
            formatted = "".join(traceback.format_exception(*err)).rstrip()
            block.append("traceback=" + formatted)
        block.append("")
        self._blocks.extend(block)

    @staticmethod
    def _format_subtest_id(test, subtest) -> str:
        params = getattr(subtest, "params", None)
        if isinstance(params, dict) and params:
            details = ", ".join(f"{key}={value!r}" for key, value in params.items())
            return f"{test.id()} ({details})"
        return str(subtest)


class Stage01LoggingRunner(unittest.TextTestRunner):
    def __init__(self, *, log_path: Path, **kwargs) -> None:
        self._log_path = log_path
        super().__init__(**kwargs)

    def _makeResult(self):
        return Stage01LoggingResult(
            self.stream,
            self.descriptions,
            self.verbosity,
            log_path=self._log_path,
        )


def build_suite() -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module_name in STAGE_01_MODULES:
        suite.addTests(loader.loadTestsFromName(module_name))
    return suite


def main() -> int:
    log_path = Path(__file__).resolve().parent / "tests_log_OfBestVar"
    runner = Stage01LoggingRunner(verbosity=2, log_path=log_path)
    result = runner.run(build_suite())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
