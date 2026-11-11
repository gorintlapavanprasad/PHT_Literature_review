"""Issue #15 (and the OpenAI side of #14): never send `temperature` to models that reject it."""
import json
from types import SimpleNamespace

import pytest
from openai import BadRequestError

from llm.gemini_llm import GeminiLLM, GeminiSettings, gemini_accepts_sampling
from llm.openai_llm import OpenAILLM, OpenAISettings

GOOD = json.dumps({"answer": 3, "confidence": 0.8, "evidence": ["chunk_1"], "reasoning": "r"})


@pytest.mark.parametrize("model,expected", [
    ("gemini-2.5-flash", True), ("gemini-2.0-flash", True), ("gemini-1.5-pro", True),
    ("gemini-3.6-flash", False), ("gemini-3-pro", False), ("gemini-3.5-flash-lite", False), ("gemini-10-x", False),
    ("gemma-4-31b-it", True), ("something-else", True),
])
def test_gemini_version_rule(model, expected, monkeypatch):
    monkeypatch.delenv("GEMINI_SEND_TEMPERATURE", raising=False)
    assert gemini_accepts_sampling(model) is expected


def test_gemini_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_SEND_TEMPERATURE", "0")
    assert gemini_accepts_sampling("gemini-2.5-flash") is False
    monkeypatch.setenv("GEMINI_SEND_TEMPERATURE", "1")
    assert gemini_accepts_sampling("gemini-3.6-flash") is True


def _gemini(model, monkeypatch, behaviour=None):
    monkeypatch.delenv("GEMINI_SEND_TEMPERATURE", raising=False)
    llm = GeminiLLM(GeminiSettings(api_key="x", model=model, temperature=0.1))
    seen = []

    class Models:
        def generate_content(self, model, contents, config):
            seen.append(config)
            if behaviour:
                behaviour(len(seen))
            return SimpleNamespace(parsed=None, text=GOOD)

    llm._client = SimpleNamespace(models=Models())
    return llm, seen


def test_gemini_old_model_sends_temperature_new_model_does_not(monkeypatch):
    llm, seen = _gemini("gemini-2.5-flash", monkeypatch)
    assert json.loads(llm.generate("p"))["answer"] == 3 and seen[0].temperature == 0.1
    llm, seen = _gemini("gemini-3.6-flash", monkeypatch)
    assert json.loads(llm.generate("p"))["answer"] == 3 and seen[0].temperature is None


def test_gemini_retries_without_temperature_when_api_rejects_it(monkeypatch):
    def boom(n):
        if n == 1:
            raise RuntimeError("400 INVALID_ARGUMENT: temperature is not supported for this model")
    llm, seen = _gemini("gemini-2.5-flash", monkeypatch, boom)
    assert json.loads(llm.generate("p"))["answer"] == 3
    assert len(seen) == 2 and seen[0].temperature == 0.1 and seen[1].temperature is None


def _openai(responder):
    llm = OpenAILLM(OpenAISettings("k", "gpt-x", 0.1))
    seen = []

    class Comp:
        def create(self, **kw):
            seen.append(kw)
            return responder(len(seen))

    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=Comp()))
    return llm, seen


def _ok():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=GOOD))])


def _bad_request(msg):
    import httpx
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    return BadRequestError(msg, response=resp, body=None)


def test_openai_retries_without_temperature_once_and_remembers():
    def responder(n):
        if n == 1:
            raise _bad_request("Unsupported value: 'temperature' does not support 0.1 with this model")
        return _ok()
    llm, seen = _openai(responder)
    assert json.loads(llm.generate("p"))["answer"] == 3
    assert "temperature" in seen[0] and "temperature" not in seen[1]
    llm.generate("p")                                           # remembered: no temperature on later calls
    assert "temperature" not in seen[2]


def test_openai_other_bad_requests_still_raise():
    llm, _ = _openai(lambda n: (_ for _ in ()).throw(_bad_request("model_not_found")))
    with pytest.raises(BadRequestError):
        llm.generate("p")
