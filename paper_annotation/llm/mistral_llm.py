"""Mistral (la Plateforme) implementation of BaseLLM.

Mistral exposes an OpenAI-compatible chat completions API, so this reuses the
Groq wrapper's retry/validation logic with a different base_url.
"""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from llm.groq_llm import GroqLLM


@dataclass(frozen=True)
class MistralSettings:
    """Mistral settings container."""

    api_key: str
    model_name: str
    temperature: float


class MistralLLM(GroqLLM):
    """Mistral LLM wrapper (OpenAI SDK pointed at api.mistral.ai)."""

    def __init__(self, settings: MistralSettings) -> None:
        self._client = OpenAI(api_key=settings.api_key, base_url="https://api.mistral.ai/v1")
        self._model_name = settings.model_name
        self._temperature = settings.temperature


if __name__ == "__main__":
    print("Mistral LLM ready.")
