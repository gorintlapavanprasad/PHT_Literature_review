import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import list_models as L  # noqa: E402

IDS = ["google/gemma-3-27b-it", "google/gemma-3-27b-it:free", "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-small:free"]


def test_filters():
    assert L.filter_ids(IDS, "gemma") == ["google/gemma-3-27b-it", "google/gemma-3-27b-it:free"]
    assert L.filter_ids(IDS, free=True) == ["google/gemma-3-27b-it:free", "mistralai/mistral-small:free"]
    assert L.filter_ids(IDS, "GEMMA", free=True) == ["google/gemma-3-27b-it:free"]
    assert L.filter_ids(IDS, "nothing") == []


def test_fetch_uses_bearer_header_without_leaking_and_parses_ids():
    seen = {}

    def getter(url, headers):
        seen.update(url=url, headers=headers)
        return {"data": [{"id": "a"}, {"id": "b"}]}

    assert L.fetch_ids("groq", {"GROQ_API_KEY": "secret-value"}, getter) == ["a", "b"]
    assert seen["url"] == "https://api.groq.com/openai/v1/models" and seen["headers"]["Authorization"] == "Bearer secret-value"


def test_openrouter_needs_no_key():
    got = L.fetch_ids("openrouter", {}, lambda url, h: {"data": [{"id": "x:free"}]})
    assert got == ["x:free"]


def test_missing_key_and_unknown_provider_exit_cleanly():
    with pytest.raises(SystemExit) as e:
        L.fetch_ids("groq", {}, lambda *a: {})
    assert "GROQ_API_KEY" in str(e.value) and "secret" not in str(e.value)
    with pytest.raises(SystemExit):
        L.fetch_ids("nonsense", {}, lambda *a: {})
