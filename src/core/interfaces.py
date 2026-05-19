from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import RetrievedChunk


class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, prompt: str) -> str:
        raise NotImplementedError


class KnowledgeBaseSearcher(ABC):
    @abstractmethod
    def search(self, query: str, topic_id: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError
