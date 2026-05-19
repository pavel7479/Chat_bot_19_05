from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _failed_scenarios_from_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    failed: list[str] = []
    pattern = re.compile(
        r"scenario=([^|]+)\s+\|\s+turn=\d+\s+\|\s+postcheck query=.*\|\s+topics=.*\|\s+classifier_ok=(True|False)\s+\|\s+response_ok=(True|False)\s+\|"
    )
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        scenario = match.group(1).strip()
        classifier_ok = match.group(2) == "True"
        response_ok = match.group(3) == "True"
        if not (classifier_ok and response_ok) and scenario not in failed:
            failed.append(scenario)
    return failed


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    log_path = project_root / "logs/tests.log"
    failed = _failed_scenarios_from_log(log_path)
    if not failed:
        print("Нет проваленных сценариев в logs/tests.log")
        return 0

    exit_code = 0
    for scenario in failed:
        test_id = f"tests.test_chat_bot.ChatBotE2ETests.test_chat_bot_{scenario}"
        print(f"[RUN] {test_id}")
        result = subprocess.run(
            [str(project_root.parent / ".venv/bin/python"), "-m", "unittest", "-v", test_id],
            cwd=str(project_root),
        )
        if result.returncode != 0:
            exit_code = result.returncode
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

# cd /root/project/Chat_bot && /root/project/.venv/bin/python tests/run_failed_chatbot_tests.py
