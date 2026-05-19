from __future__ import annotations

import json
import os

import requests

from src.config.schema import LLMConfig
from src.core.exceptions import LLMProviderError
from src.core.interfaces import LLMProvider

class GemmaOllamaProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        self._clear_proxy_env()
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model_name
        self.timeout = config.timeout
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    @staticmethod
    def _clear_proxy_env() -> None:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(key, None)

    def generate_json(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": self.max_tokens,
            },
        }
        return self._stream_generate(payload)

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return self._stream_generate(payload)

    def _stream_generate(self, payload: dict[str, object]) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                proxies={"http": None, "https": None},
                stream=True,
            )
            response.raise_for_status()

            chunks: list[str] = []
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    chunks.append(data["response"])
                if data.get("done"):
                    break
            return "".join(chunks)
        except Exception as error:
            raise LLMProviderError(f"Gemma/Ollama generate failed: {error}") from error
