"""Registry of Claude model versions (issue #10).

IDs and list prices (USD per million tokens) come from Anthropic's model table cached 2026-11-03 - CHECK
https://platform.claude.com/docs/en/about-claude/models/overview before a big run; run
`python scripts/smoke_models.py` to confirm every ID still answers.

Per-model facts that change how a request must be built:
  * no custom `temperature` is ever sent (removed / rejected on the 4.7+ and 5.x models);
  * `output_config.effort` (our cost/quality lever) is rejected by Haiku 4.5 -> `supports_effort=False`;
  * Fable always thinks (thinking param omitted - we never send it) and needs 30-day data retention.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str            # fable | opus | sonnet | haiku
    generation: str        # e.g. "5.5"
    in_price: float        # USD / million input tokens (prompts up to 100K tokens)
    out_price: float       # USD / million output tokens
    supports_effort: bool = True
    current: bool = False  # part of the default `claude` alias (the newest of each tier)
    note: str = ""


CLAUDE_MODELS: list[ModelSpec] = [
    ModelSpec("claude-fable-5-2", "fable", "5.2", 8.00, 40.00, note="most capable; thinking always on; 30-day retention required"),
    ModelSpec("claude-fable-5-1", "fable", "5.1", 10.00, 50.00, note="thinking always on; 30-day retention required"),
    ModelSpec("claude-fable-5", "fable", "5", 10.00, 50.00, note="thinking always on; 30-day retention required"),
    ModelSpec("claude-opus-5-5", "opus", "5.5", 4.00, 20.00, current=True),
    ModelSpec("claude-opus-5", "opus", "5", 5.00, 25.00),
    ModelSpec("claude-opus-4-8", "opus", "4.8", 5.00, 25.00),
    ModelSpec("claude-opus-4-7", "opus", "4.7", 5.00, 25.00),
    ModelSpec("claude-opus-4-6", "opus", "4.6", 5.00, 25.00),
    ModelSpec("claude-sonnet-5-5", "sonnet", "5.5", 2.00, 10.00, current=True),
    ModelSpec("claude-sonnet-5", "sonnet", "5", 2.00, 10.00),
    ModelSpec("claude-sonnet-4-6", "sonnet", "4.6", 3.00, 15.00),
    ModelSpec("claude-haiku-5-5", "haiku", "5.5", 0.10, 0.50, current=True, note="price rises beyond 100K-token prompts"),
    ModelSpec("claude-haiku-4-5", "haiku", "4.5", 1.00, 5.00, supports_effort=False, note="effort parameter is rejected"),
]

BY_ID: dict[str, ModelSpec] = {m.id: m for m in CLAUDE_MODELS}


def claude_ids() -> list[str]:
    """Every registered version, newest tier first."""
    return [m.id for m in CLAUDE_MODELS]


def claude_current_ids() -> list[str]:
    """The newest model of each of the Opus / Sonnet / Haiku tiers (default `claude` alias)."""
    return [m.id for m in CLAUDE_MODELS if m.current]


def spec(model_id: str) -> ModelSpec | None:
    return BY_ID.get(model_id)


def supports_effort(model_id: str) -> bool:
    """Unknown IDs are assumed modern (effort supported); only registered exceptions opt out."""
    s = BY_ID.get(model_id)
    return True if s is None else s.supports_effort
