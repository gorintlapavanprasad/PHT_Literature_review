"""Groq implementation of BaseLLM.

Groq exposes an OpenAI-compatible chat completions API, so this is essentially
the OpenAI backend with a different base_url. The openai Python SDK works
directly by passing `base_url="https://api.groq.com/openai/v1"`.

Groq's free tier is generous (6k tokens/min, 500k tokens/day on Llama 3.3 70B)
and inference is extremely fast — typical responses in 1-2 seconds vs.
5-15 seconds on HuggingFace Inference for the same model.
"""
from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass
from typing import Dict, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError, BadRequestError
from pydantic import BaseModel, Field, ValidationError

from llm.base import BaseLLM


class LLMResponse(BaseModel):
    """Validated LLM response schema."""

    answer: int | str = Field(..., description="1-5 or 'Unknown'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str


@dataclass(frozen=True)
class GroqSettings:
    """Groq settings container."""

    api_key: str
    model_name: str
    temperature: float


class GroqLLM(BaseLLM):
    """Groq LLM wrapper built on the openai SDK with Groq's base_url."""

    def __init__(self, settings: GroqSettings) -> None:
        # The openai client is generic — pointing it at Groq's OpenAI-compatible
        # endpoint means everything else (retries, streaming, JSON mode) just works.
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model_name = settings.model_name
        self._temperature = settings.temperature

    def _parse(self, content: str) -> Optional[Dict]:
        try:
            data = json.loads(content)
            validated = LLMResponse(**data)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError):
            return None

    def _call_api(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        max_attempts = 4
        base_delay = 1.0
        content = ""

        for attempt in range(1, max_attempts + 1):
            try:
                content = self._call_api(prompt)
                break
            except BadRequestError:
                raise
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1)) + random.random()
                time.sleep(delay)
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status < 500:
                    raise
                if attempt == max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1)) + random.random()
                time.sleep(delay)

        parsed = self._parse(content)
        if parsed is None:
            # One retry in case the model produced malformed JSON
            retry_content = self._call_api(prompt)
            parsed = self._parse(retry_content)

        if parsed is None:
            parsed = {
                "answer": "Unknown",
                "confidence": 0.0,
                "evidence": [],
                "reasoning": "Invalid JSON response from model.",
            }

        parsed["latency_seconds"] = time.perf_counter() - start
        return json.dumps(parsed, ensure_ascii=False)


if __name__ == "__main__":
    print("Groq LLM ready.")
