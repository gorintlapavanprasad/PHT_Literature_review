"""Issue #11: smoke_models.py - skip without keys, report failures clearly, never leak keys."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import smoke_models as S  # noqa: E402


class GoodLLM:
    def generate(self, prompt):
        return json.dumps({"answer": 3, "confidence": 0.9, "evidence": [], "reasoning": "ok"})


class WrongLLM:
    def generate(self, prompt):
        return json.dumps({"answer": "Unknown", "confidence": 0.0, "evidence": [], "reasoning": "x"})


def provider_of(name):
    return {"claude-x": "anthropic", "gpt-x": "openai", "bad-model": "anthropic", "wrong": "anthropic"}[name]


def build(name, cfg):
    if name == "bad-model":
        raise RuntimeError("404 model: not_found_error " + "sk-ant-" + "api03-ABCDEFGHIJKLMNOP1234567890")
    return WrongLLM() if name == "wrong" else GoodLLM()


def test_missing_keys_skip_and_name_the_variable():
    r = S.run_smoke(["claude-x", "gpt-x"], {}, build, provider_of)
    assert [x["status"] for x in r] == ["SKIP", "SKIP"]
    assert "ANTHROPIC_API_KEY" in r[0]["detail"] and "OPENAI_API_KEY" in r[1]["detail"]


def test_ok_fail_and_key_never_leaked():
    env = {"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": " "}              # blank counts as missing
    r = {x["model"]: x for x in S.run_smoke(["claude-x", "gpt-x", "bad-model", "wrong"], env, build, provider_of)}
    assert r["claude-x"]["status"] == "OK" and r["gpt-x"]["status"] == "SKIP"
    assert r["bad-model"]["status"] == "FAIL" and "not_found_error" in r["bad-model"]["detail"]
    assert "ABCDEFGHIJKLMNOP1234567890" not in r["bad-model"]["detail"]  # key masked
    assert r["wrong"]["status"] == "FAIL" and "unexpected answer" in r["wrong"]["detail"]


def test_table_summary_counts():
    res = [{"model": "a", "status": "OK", "seconds": 1.0}, {"model": "b", "status": "SKIP", "detail": "set X"},
           {"model": "c", "status": "FAIL", "detail": "boom"}]
    t = S.format_table(res)
    assert "1 ok, 1 failed, 1 skipped" in t and "set X" in t


def test_sanitize_masks_common_key_shapes():
    for raw in ("sk-" + "proj-ABCDEFGH12345678", "AIza" + "SyABCDEFGHIJKLMN", "gsk_" + "abcdefghijklmn", "hf_" + "abcdefghijklmn"):
        assert raw[8:] not in S._sanitize(f"auth failed for {raw}")


def test_dry_run_makes_no_api_calls_and_exit_zero(monkeypatch, capsys):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert S.main(["--dry-run", "--models", "claude"]) == 0
    out = capsys.readouterr().out
    assert "NEEDS ANTHROPIC_API_KEY" in out and "claude-haiku-5-5" in out


def test_main_exit_code_is_zero_when_everything_skipped(monkeypatch, capsys):
    for k in ("ANTHROPIC_API_KEY",):
        monkeypatch.delenv(k, raising=False)
    assert S.main(["--models", "claude-haiku-5-5"]) == 0
    assert "1 skipped" in capsys.readouterr().out
