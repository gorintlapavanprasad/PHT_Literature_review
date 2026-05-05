"""Hugging Face Inference API implementation of BaseLLM."""
from __future__ import annotations

import json
import re
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field, ValidationError

from llm.base import BaseLLM


class LLMResponse(BaseModel):
    """Validated LLM response schema."""

    answer: int | str = Field(..., description="1-5 or 'Unknown'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str


@dataclass(frozen=True)
class HuggingFaceSettings:
    """Hugging Face settings container."""

    token: str
    model_name: str
    temperature: float
    timeout_seconds: int = 120


class HuggingFaceLLM(BaseLLM):
    """Hugging Face Inference API wrapper using huggingface_hub InferenceClient."""

    def __init__(self, settings: HuggingFaceSettings) -> None:
        self._model_name = settings.model_name
        self._temperature = settings.temperature
        self._client = InferenceClient(
            model=settings.model_name,
            token=settings.token,
            timeout=settings.timeout_seconds,
        )

    def _parse(self, content: str) -> Optional[Dict]:
        """Try to extract a valid LLMResponse JSON from model output.

        Handles: raw JSON, JSON in ```json``` fences, and JSON embedded in prose.
        """
        def _validate(text: str) -> Optional[Dict]:
            try:
                data = json.loads(text)
                validated = LLMResponse(**data)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError):
                return None

        # 1. Try raw content directly.
        result = _validate(content)
        if result is not None:
            return result

        # 2. Extract from ```json ... ``` or ``` ... ``` fences.
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fence_match:
            result = _validate(fence_match.group(1))
            if result is not None:
                return result

        # 3. Find the outermost {...} block in the text.
        brace_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if brace_match:
            result = _validate(brace_match.group(1))
            if result is not None:
                return result

        return None

    def _call_api(self, prompt: str) -> str:
        """Call the HuggingFace Inference API via InferenceClient and return the assistant message content."""
        response = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=1024,
        )
        choices = getattr(response, "choices", [])
        if choices:
            message = getattr(choices[0], "message", None)
            if message:
                return str(getattr(message, "content", "") or "").strip()
        return ""

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        max_attempts = 5
        base_delay = 1.0
        last_api_error: Optional[Exception] = None
        last_parse_failed = False

        for attempt in range(1, max_attempts + 1):
            try:
                content = self._call_api(prompt)
            except Exception as exc:
                last_api_error = exc
                last_parse_failed = False
                if attempt == max_attempts:
                    break
                delay = base_delay * (2 ** (attempt - 1)) + random.random()
                time.sleep(delay)
                continue

            last_api_error = None
            parsed = self._parse(content)
            if parsed is None:
                last_parse_failed = True
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1)) + random.random()
                    time.sleep(delay)
                    continue
            else:
                last_parse_failed = False
                parsed["latency_seconds"] = time.perf_counter() - start
                return json.dumps(parsed, ensure_ascii=False)

        if last_api_error is not None:
            reason = (
                f"API error after {max_attempts} attempts: "
                f"{type(last_api_error).__name__}: {last_api_error}"
            )
            mode = "api_failure"
        else:
            reason = f"Model returned unparseable output after {max_attempts} attempts."
            mode = "parse_failure"

        fallback = {
            "answer": "Unknown",
            "confidence": 0.0,
            "evidence": [],
            "decision_mode": mode,
            "reasoning": reason,
            "latency_seconds": time.perf_counter() - start,
        }
        return json.dumps(fallback, ensure_ascii=False)
