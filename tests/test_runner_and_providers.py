"""Parallel runner, run_bounded ordering, and the opt-in Claude / OpenAI-compatible providers."""
import dataclasses
import json
import time
from types import SimpleNamespace

import pytest

import config
import main as pipeline_main
from pipeline.concurrency import run_bounded
from pipeline.runner import run_jobs


# ---------------------------------------------------------------- runner
def test_run_bounded_preserves_input_order():
    def slow_first(x):
        time.sleep(0.2 if x == 0 else 0.0)
        return x
    assert run_bounded(range(8), slow_first, max_workers=4, log_semaphore=False) == list(range(8))


def test_providers_run_in_parallel_with_per_provider_cap():
    jobs = [(i, p) for i in range(6) for p in ("a", "b", "c")]

    def work(job):
        time.sleep(0.05)

    t0 = time.perf_counter()
    s = run_jobs(jobs, work, provider_of=lambda j: j[1], per_provider_limit=2)
    wall = time.perf_counter() - t0
    assert len(s.succeeded) == 18 and not s.failed
    assert all(v <= 2 for v in s.peak_concurrency.values())          # cap respected
    assert max(s.peak_concurrency.values()) == 2                     # and actually used
    assert wall < 18 * 0.05 * 0.6                                    # much faster than sequential (0.9 s)


def test_failed_job_is_reported_and_does_not_stop_others():
    def work(job):
        if job == 3:
            raise RuntimeError("boom")

    s = run_jobs(range(8), work, provider_of=lambda j: "p", per_provider_limit=3)
    assert len(s.succeeded) == 7 and [j for j, _ in s.failed] == [3] and "boom" in s.failed[0][1]


def test_runner_empty_jobs():
    s = run_jobs([], lambda j: None, provider_of=lambda j: "p")
    assert s.succeeded == [] and s.failed == []


# ---------------------------------------------------------------- config / model selection
def test_default_models_exclude_opt_in_claude():
    assert not set(config.EXTRA_MODELS["anthropic"]) & set(config.all_models())
    assert pipeline_main._resolve_models(None, config.all_models(), config.selectable_models()) == config.all_models()


def test_claude_alias_and_exact_names():
    sel = config.selectable_models()
    assert set(pipeline_main._resolve_models(["claude"], config.all_models(), sel)) == {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}
    assert pipeline_main._resolve_models(["claude-opus-5-5"], config.all_models(), sel) == ["claude-opus-5-5"]
    with pytest.raises(SystemExit):
        pipeline_main._resolve_models(["claude-opus-5-5"], config.all_models())   # not in the default list


def test_each_model_gets_its_own_results_dir_and_unknown_model_never_falls_back_to_openai():
    dirs = {m: config.model_output_dir(m) for m in config.selectable_models()}
    assert len(set(dirs.values())) == len(dirs)                      # no two models share a folder
    assert dirs["claude-sonnet-5-5"] == "anthropic_claude_sonnet_5_5"
    with pytest.raises(ValueError):                                  # regression: used to return "openai"
        config.model_output_dir("some-new-model")


def test_build_llm_requires_keys():
    cfg = dataclasses.replace(config.load_config(), anthropic_api_key="", compat_api_key="", compat_base_url="")
    with pytest.raises(EnvironmentError):
        pipeline_main._build_llm("claude-sonnet-5-5", cfg)


# ---------------------------------------------------------------- providers (no network)
GOOD = json.dumps({"answer": 3, "confidence": 0.7, "evidence": ["chunk_1"], "reasoning": "r"})


def _resp(text, stop="end_turn", thinking=False):
    content = ([SimpleNamespace(type="thinking", thinking="hmm")] if thinking else []) + \
              [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason=stop)


def _claude(responses):
    from llm import anthropic_llm
    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            return responses[min(len(calls) - 1, len(responses) - 1)]

    llm = anthropic_llm.AnthropicLLM(anthropic_llm.AnthropicSettings("k", "claude-sonnet-5-5", 0.1))
    llm._client = SimpleNamespace(messages=FakeMessages())
    return llm, calls


def test_claude_request_shape_has_no_temperature_and_uses_effort():
    llm, calls = _claude([_resp(GOOD, thinking=True)])
    out = json.loads(llm.generate("prompt"))
    assert out["answer"] == 3 and out["latency_seconds"] >= 0
    kw = calls[0]
    assert "temperature" not in kw and kw["model"] == "claude-sonnet-5-5"
    assert kw["output_config"] == {"effort": "medium"} and kw["max_tokens"] >= 4096


@pytest.mark.parametrize("text", ["```json\n" + GOOD + "\n```", "Here you go: " + GOOD + " thanks"])
def test_claude_tolerates_fenced_or_wrapped_json(text):
    llm, _ = _claude([_resp(text)])
    assert json.loads(llm.generate("p"))["answer"] == 3


def test_claude_reasks_once_on_bad_json_then_gives_unknown():
    llm, calls = _claude([_resp("garbage"), _resp(GOOD)])
    assert json.loads(llm.generate("p"))["answer"] == 3 and len(calls) == 2
    llm, calls = _claude([_resp("garbage")])
    out = json.loads(llm.generate("p"))
    assert out["answer"] == "Unknown" and len(calls) == 2


def test_claude_refusal_becomes_honest_unknown():
    llm, calls = _claude([_resp("", stop="refusal")])
    out = json.loads(llm.generate("p"))
    assert out["answer"] == "Unknown" and "declined" in out["reasoning"]


def test_compat_provider_parses_and_has_no_json_mode():
    from llm.compat_llm import CompatLLM, CompatSettings
    seen = []

    class Comp:
        def create(self, **kw):
            seen.append(kw)
            msg = SimpleNamespace(content="```json\n" + GOOD + "\n```")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    llm = CompatLLM(CompatSettings("k", "http://localhost:1/v1", "google/gemma-3-27b-it", 0.1))
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=Comp()))
    assert json.loads(llm.generate("p"))["answer"] == 3
    assert "response_format" not in seen[0] and seen[0]["model"] == "google/gemma-3-27b-it"


def test_json_utils_validation_rejects_bad_confidence():
    from llm.json_utils import parse_validated
    assert parse_validated(json.dumps({"answer": 3, "confidence": 1.5, "evidence": [], "reasoning": "r"})) is None
    assert parse_validated("no json") is None
