from __future__ import annotations

from src.config.schema import LLMConfig
from src.core.exceptions import ConfigurationError
from src.core.interfaces import LLMProvider
from src.llm.gemma_ollama_provider import GemmaOllamaProvider


class LLMProviderFactory:
    @staticmethod
    def create(config: LLMConfig) -> LLMProvider:
        if config.provider == "gemma_ollama":
            return GemmaOllamaProvider(config)
        raise ConfigurationError(f"Unknown llm.provider: {config.provider}")
