"""Generic OpenAI-compatible provider for newer open models (Llama, Mistral, Gemma, ...).

Point it at ANY OpenAI-compatible host (Groq, Together, OpenRouter, Fireworks, a local Ollama/vLLM):

    COMPAT_BASE_URL=https://openrouter.ai/api/v1
    COMPAT_API_KEY=...
    COMPAT_MODELS=google/gemma-3-27b-it,meta-llama/llama-4-maverick,mistralai/mistral-large-2411

Each model's results go to their own folder (data/results/compat_<slug>/), so runs never mix.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from openai import OpenAI

from llm.base import BaseLLM
from llm.json_utils import UNKNOWN_RESPONSE, parse_validated


@dataclass(frozen=True)
class CompatSettings:
    """Settings for an OpenAI-compatible endpoint."""

    api_key: str
    base_url: str
    model_name: str
    temperature: float


class CompatLLM(BaseLLM):
    """OpenAI-compatible chat model with tolerant JSON parsing (no JSON-mode dependency)."""

    def __init__(self, settings: CompatSettings) -> None:
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=6, timeout=300.0)
        self._model_name = settings.model_name
        self._temperature = settings.temperature

    def _call_api(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model_name,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        parsed = parse_validated(self._call_api(prompt))
        if parsed is None:
            parsed = parse_validated(self._call_api(prompt))
        if parsed is None:
            parsed = dict(UNKNOWN_RESPONSE)
        parsed["latency_seconds"] = time.perf_counter() - start
        return json.dumps(parsed, ensure_ascii=False)


if __name__ == "__main__":
    print("Compat LLM ready.")
