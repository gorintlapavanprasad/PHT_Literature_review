"""LLM base abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract LLM interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a raw JSON string response from a prompt."""
        raise NotImplementedError
