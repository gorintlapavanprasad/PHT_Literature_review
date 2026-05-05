"""Gemini implementation of BaseLLM."""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from llm.base import BaseLLM


class LLMResponse(BaseModel):
    """Validated LLM response schema."""

    answer: int | str = Field(..., description="1-5 or 'Unknown'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str


@dataclass(frozen=True)
class GeminiSettings:
    """Gemini settings container."""

    api_key: str
    model: str = "gemini-2.0-flash-pro"
    temperature: float = 0.1
    max_tokens: int | None = None


class GeminiLLM(BaseLLM):
    """Gemini LLM wrapper."""

    def __init__(self, settings: GeminiSettings) -> None:
        self.settings = settings
        self._client = genai.Client(api_key=settings.api_key)

    def _validate(self, payload: dict) -> Optional[Dict]:
        try:
            validated = LLMResponse(**payload)
            return validated.model_dump()
        except ValidationError:
            return None

    def _normalize_payload(self, payload: dict) -> dict:
        normalized = dict(payload)

        answer = normalized.get("answer", "Unknown")
        if isinstance(answer, str):
            stripped = answer.strip()
            if stripped.isdigit():
                answer = int(stripped)
            elif stripped.lower() == "unknown":
                answer = "Unknown"
        if isinstance(answer, int) and not (1 <= answer <= 5):
            answer = "Unknown"
        normalized["answer"] = answer

        confidence = normalized.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        normalized["confidence"] = confidence

        evidence = normalized.get("evidence", [])
        if evidence is None:
            evidence = []
        elif isinstance(evidence, str):
            evidence = [evidence] if evidence.strip() else []
        elif not isinstance(evidence, list):
            evidence = []
        normalized["evidence"] = [str(item) for item in evidence if str(item).strip()]

        reasoning = normalized.get("reasoning", "")
        if reasoning is None:
            reasoning = ""
        normalized["reasoning"] = str(reasoning)

        return normalized

    def _extract_json_object(self, content: str) -> Optional[dict]:
        # Gemini may return JSON inside code fences or with leading/trailing text.
        text = (content or "").strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None

        return None

    def _parse(self, content: str) -> Optional[Dict]:
        try:
            data = self._extract_json_object(content)
            if data is None:
                return None
            normalized = self._normalize_payload(data)
            return self._validate(normalized)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return None

    def _call_gemini(self, prompt: str) -> str:
        generation_config = types.GenerateContentConfig(
            temperature=self.settings.temperature,
            response_mime_type="application/json",
            response_schema=LLMResponse,
        )
        if self.settings.max_tokens is not None:
            generation_config.max_output_tokens = self.settings.max_tokens

        response = self._client.models.generate_content(
            model=self.settings.model,
            contents=prompt,
            config=generation_config,
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        return response.text or ""

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        max_attempts = 3
        base_delay = 1.0
        content = ""

        for attempt in range(1, max_attempts + 1):
            try:
                content = self._call_gemini(prompt)
                break
            except Exception:
                if attempt == max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1)) + random.random()
                time.sleep(delay)

        parsed = self._parse(content)
        if parsed is None:
            retry_content = self._call_gemini(prompt)
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
    print("Gemini LLM ready.")
