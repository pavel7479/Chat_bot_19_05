from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config.schema import LoggingConfig


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            payload["extra"] = record.extra_data
        return json.dumps(payload, ensure_ascii=False)


class StructuredLoggerFactory:
    @staticmethod
    def create(name: str, config: LoggingConfig, project_root: Path) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
        logger.handlers.clear()
        logger.propagate = False

        file_path = project_root / config.file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = JsonFormatter() if config.json_format else logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if config.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        StructuredLoggerFactory._configure_related_logger(
            parent_name=name,
            suffix="turns",
            level=logger.level,
            file_path=StructuredLoggerFactory._related_file_path(file_path, "turns"),
            formatter=formatter,
        )
        StructuredLoggerFactory._configure_related_logger(
            parent_name=name,
            suffix="failures",
            level=logger.level,
            file_path=StructuredLoggerFactory._related_file_path(file_path, "failures"),
            formatter=formatter,
        )

        return logger

    @staticmethod
    def get_related_logger(parent_name: str, suffix: str) -> logging.Logger:
        return logging.getLogger(f"{parent_name}.{suffix}")

    @staticmethod
    def _related_file_path(file_path: Path, suffix: str) -> Path:
        return file_path.with_name(f"{file_path.stem}_{suffix}{file_path.suffix}")

    @staticmethod
    def _configure_related_logger(
        *,
        parent_name: str,
        suffix: str,
        level: int,
        file_path: Path,
        formatter: logging.Formatter,
    ) -> None:
        logger = logging.getLogger(f"{parent_name}.{suffix}")
        logger.setLevel(level)
        logger.handlers.clear()
        logger.propagate = False

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def log_event(logger: logging.Logger, event: str, **kwargs: object) -> None:
    logger.info(event, extra={"extra_data": kwargs})
