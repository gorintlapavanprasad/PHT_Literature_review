"""Tolerant extraction/validation of the model's JSON answer (shared by newer providers)."""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError


class LLMResponse(BaseModel):
    """Validated LLM response schema."""

    answer: int | str = Field(..., description="1-5, 'Unknown' or 'Not Applicable'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str


def content_to_text(content) -> str:
    """Chat 'content' as plain text. Some APIs (e.g. newer Mistral models) return a LIST of parts such as
    [{"type": "text", "text": "..."}] instead of a string; anything that is not text (thinking parts) is dropped."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text":
                parts.append(str(p.get("text", "")))
            elif getattr(p, "type", "text") == "text" and getattr(p, "text", None) is not None:
                parts.append(str(p.text))
        return "".join(parts)
    return str(content)


def extract_json_object(text: str) -> Optional[dict]:
    """Return the first JSON object in `text` (raw, ```json fenced, or embedded in prose)."""
    text = (text or "").strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            data = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_validated(text: str) -> Optional[dict]:
    """Extract + validate against LLMResponse; None if unusable."""
    data = extract_json_object(text)
    if data is None:
        return None
    try:
        return LLMResponse(**data).model_dump()
    except ValidationError:
        return None


UNKNOWN_RESPONSE = {
    "answer": "Unknown",
    "confidence": 0.0,
    "evidence": [],
    "reasoning": "Invalid JSON response from model.",
}
