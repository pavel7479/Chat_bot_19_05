from __future__ import annotations

from pathlib import Path

import yaml

from src.config.schema import (
    ApiConfig,
    AppConfig,
    Config,
    LLMConfig,
    LoggingConfig,
    PathsConfig,
    RetrievalConfig,
    SessionConfig,
)
from src.core.exceptions import ConfigurationError


class ConfigLoader:
    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)

    def load(self) -> Config:
        if not self._config_path.exists():
            raise ConfigurationError(f"Config not found: {self._config_path}")

        with self._config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        try:
            return Config(
                app=AppConfig(**raw["app"]),
                llm=LLMConfig(**raw["llm"]),
                paths=PathsConfig(**raw["paths"]),
                retrieval=RetrievalConfig(**raw["retrieval"]),
                session=SessionConfig(**raw["session"]),
                logging=LoggingConfig(**raw["logging"]),
                api=ApiConfig(**raw["api"]),
            )
        except KeyError as error:
            raise ConfigurationError(f"Missing config section: {error}") from error
        except TypeError as error:
            raise ConfigurationError(f"Invalid config structure: {error}") from error
