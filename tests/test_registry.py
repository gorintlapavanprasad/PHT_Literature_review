"""Issue #10: every Claude version is registered, selectable, and requested correctly."""
import json
from types import SimpleNamespace

import pytest

import config
import main as pipeline_main
from llm import registry as R
from llm.anthropic_llm import AnthropicLLM, AnthropicSettings

EXPECTED = {
    "claude-fable-5-2", "claude-fable-5-1", "claude-fable-5",
    "claude-opus-5-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6",
    "claude-haiku-5-5", "claude-haiku-4-5",
}


def test_registry_covers_every_documented_version_without_duplicates():
    ids = R.claude_ids()
    assert set(ids) == EXPECTED and len(ids) == len(set(ids)) == 13
    assert all(i.startswith("claude-") and not any(c.isdigit() and len(i.split("-")[-1]) == 8 for c in i) for i in ids)


def test_prices_are_sane_and_ordered_by_tier():
    for m in R.CLAUDE_MODELS:
        assert 0 < m.in_price < m.out_price
    assert R.spec("claude-haiku-5-5").in_price < R.spec("claude-sonnet-5-5").in_price < R.spec("claude-opus-5-5").in_price


def test_current_alias_is_newest_of_each_tier():
    assert set(R.claude_current_ids()) == {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}


def test_effort_support_flags():
    assert not R.supports_effort("claude-haiku-4-5")
    assert all(R.supports_effort(i) for i in EXPECTED - {"claude-haiku-4-5"})
    assert R.supports_effort("claude-some-future-model")          # unknown -> assume modern


def test_all_versions_selectable_and_aliases_resolve():
    sel = config.selectable_models()
    assert EXPECTED <= set(sel)
    assert not EXPECTED & set(config.all_models())                # never part of the default run
    assert set(pipeline_main._resolve_models(["claude-all"], config.all_models(), sel)) == EXPECTED
    assert set(pipeline_main._resolve_models(["claude"], config.all_models(), sel)) == set(R.claude_current_ids())
    assert pipeline_main._resolve_models(["claude-fable-5-1"], config.all_models(), sel) == ["claude-fable-5-1"]


def test_each_version_has_its_own_results_folder():
    dirs = [config.model_output_dir(i) for i in sorted(EXPECTED)]
    assert len(set(dirs)) == 13 and all(d.startswith("anthropic_claude_") for d in dirs)


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_request_shape_per_version(model):
    calls = []
    good = json.dumps({"answer": 2, "confidence": 0.5, "evidence": [], "reasoning": "r"})

    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=good)], stop_reason="end_turn")

    llm = AnthropicLLM(AnthropicSettings("k", model, 0.1))
    llm._client = SimpleNamespace(messages=Msgs())
    assert json.loads(llm.generate("p"))["answer"] == 2
    kw = calls[0]
    assert kw["model"] == model and "temperature" not in kw and "thinking" not in kw
    if model == "claude-haiku-4-5":
        assert "output_config" not in kw               # would be a 400
    else:
        assert kw["output_config"] == {"effort": "medium"}
