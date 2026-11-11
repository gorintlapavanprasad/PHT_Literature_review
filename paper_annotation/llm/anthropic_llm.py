"""Claude (Anthropic) implementation of BaseLLM.

Notes for the Claude 5.x family (claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5):
  * custom `temperature` is not accepted (400 / removed), so none is sent - determinism is
    handled by the pipeline running each (paper, question) once and storing the record;
  * thinking runs adaptively by default - depth is controlled with `output_config.effort`
    (env ANTHROPIC_EFFORT, default "medium");
  * a safety classifier can end a turn with stop_reason "refusal" - that is recorded as an
    honest Unknown instead of crashing the run.
The SDK retries 408/409/429/5xx and connection errors itself (max_retries below).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import anthropic

from llm.base import BaseLLM
from llm.json_utils import UNKNOWN_RESPONSE, parse_validated
from llm.registry import supports_effort


@dataclass(frozen=True)
class AnthropicSettings:
    """Anthropic settings container."""

    api_key: str
    model_name: str
    temperature: float = 0.0   # accepted for interface parity; deliberately NOT sent to the API
    effort: str = ""           # "low" | "medium" | "high" | "xhigh" | "max"; "" -> env / medium
    max_tokens: int = 0        # 0 -> env / 16000


class AnthropicLLM(BaseLLM):
    """Claude wrapper built on the official anthropic SDK."""

    def __init__(self, settings: AnthropicSettings) -> None:
        self._client = anthropic.Anthropic(api_key=settings.api_key, max_retries=6, timeout=300.0)
        self._model_name = settings.model_name
        self._effort = settings.effort or os.getenv("ANTHROPIC_EFFORT", "medium")
        self._max_tokens = settings.max_tokens or int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))

    def _call_api(self, prompt: str) -> str:
        kwargs = {}
        if supports_effort(self._model_name):       # Haiku 4.5 rejects output_config.effort
            kwargs["output_config"] = {"effort": self._effort}
        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            return json.dumps({**UNKNOWN_RESPONSE, "reasoning": "Model declined to answer (safety refusal)."})
        # Thinking blocks (if any) are separate content blocks; only the text block is the answer.
        return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        parsed = parse_validated(self._call_api(prompt))
        if parsed is None:                       # one re-ask on malformed JSON
            parsed = parse_validated(self._call_api(prompt))
        if parsed is None:
            parsed = dict(UNKNOWN_RESPONSE)
        parsed["latency_seconds"] = time.perf_counter() - start
        return json.dumps(parsed, ensure_ascii=False)


if __name__ == "__main__":
    print("Anthropic LLM ready.")
