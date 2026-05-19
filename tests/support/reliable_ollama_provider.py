from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests

from src.config.schema import LLMConfig
from src.core.exceptions import LLMProviderError
from src.core.interfaces import LLMProvider


@dataclass(slots=True)
class TestLLMRuntimeConfig:
    per_request_timeout_s: int = 35
    max_retries: int = 2
    retry_backoff_s: float = 1.0


class ReliableOllamaTestProvider(LLMProvider):
    """
    Test-only provider: non-stream JSON requests with retry and strict timeout.
    Keeps production provider untouched.
    """

    def __init__(self, llm_config: LLMConfig, runtime: TestLLMRuntimeConfig) -> None:
        self.base_url = llm_config.base_url.rstrip("/")
        self.model = llm_config.model_name
        self.temperature = llm_config.temperature
        self.max_tokens = llm_config.max_tokens
        self.timeout_s = runtime.per_request_timeout_s
        self.max_retries = max(runtime.max_retries, 0)
        self.retry_backoff_s = max(runtime.retry_backoff_s, 0.0)

    def generate_json(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": self.max_tokens,
            },
        }
        return self._request(payload)

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        return self._request(payload)

    def _request(self, payload: dict[str, object]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout_s,
                    proxies={"http": None, "https": None},
                    stream=False,
                ) as response:
                    response.raise_for_status()
                    data = response.json()
                    value = data.get("response", "")
                    if isinstance(value, str):
                        return value
                    return json.dumps(value, ensure_ascii=False)
            except Exception as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (attempt + 1))

        raise LLMProviderError(f"Reliable test provider failed: {last_error}") from last_error
