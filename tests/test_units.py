"""Unit tests: answer normalisation, response parsing, chunker, retriever, metrics, config."""
import pytest

from pipeline.annotate import _normalize_answer, _parse_llm_response, _penalize_confidence
from ingest.chunker import chunk_text
from ingest.pdf_loader import PageText
from ingest.text_cleaner import clean_text
import evaluation.metrics as metrics
import config


@pytest.mark.parametrize("raw,expected", [
    (3, 3), ("3", 3), ("3.", 3), (" 4 ", 4), ("(2)", 2), ("3.0", 3), (3.0, 3),
    ("5 - entirely asymmetrical", 5), ("1: totally fun first", 1),
    ("N/A", "Not Applicable"), ("na", "Not Applicable"), ("Not Applicable", "Not Applicable"),
    ("not applicable to this paper", "Not Applicable"),
    ("Unknown", "Unknown"), ("unknown", "Unknown"),
    (0, "Unknown"), (6, "Unknown"), (-1, "Unknown"), (True, "Unknown"), (False, "Unknown"),
    (None, "Unknown"), ("", "Unknown"), ("maybe", "Unknown"), (3.5, "Unknown"), ([3], "Unknown"),
    ("12", "Unknown"), ("3 4", 3),
])
def test_normalize_answer(raw, expected):
    assert _normalize_answer(raw) == expected


@pytest.mark.parametrize("raw,answer", [
    ('{"answer": 3}', 3),
    ('```json\n{"answer": 4}\n```', 4),
    ('```\n{"answer": 2}\n```', 2),
    ('Sure! Here you go: {"answer": 5, "confidence": 0.9} hope it helps', 5),
    ('  {"answer": 1}  ', 1),
])
def test_parse_valid_variants(raw, answer):
    assert _parse_llm_response(raw)["answer"] == answer


@pytest.mark.parametrize("raw", ["", "not json", "[1,2,3]", "{broken", None, "```json\n```"])
def test_parse_garbage_returns_unknown(raw):
    r = _parse_llm_response(raw)
    assert r["answer"] == "Unknown" and r["confidence"] == 0.0


def test_penalize_confidence():
    assert _penalize_confidence(0.8, []) == 0.4
    assert _penalize_confidence(0.8, ["chunk_1"]) == 0.8


@pytest.mark.parametrize("size,overlap,text_len", [(900, 150, 5000), (100, 10, 99), (100, 10, 100),
                                                   (100, 10, 101), (50, 49, 400), (10, 0, 95)])
def test_chunker_covers_text_without_redundant_tail(size, overlap, text_len):
    text = "".join(chr(97 + i % 26) for i in range(text_len))
    chunks = chunk_text("p", [PageText(1, text)], size, overlap)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == text_len
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start == a.char_end - overlap  # constant overlap
        assert b.char_end > a.char_end               # always progresses (no redundant tail)
    assert "".join(c.text[overlap if i else 0:] for i, c in enumerate(chunks)) == text


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("p", [PageText(1, "abc")], 10, 10)


def test_chunker_empty_and_multipage_pages():
    assert chunk_text("p", [], 100, 10) == []
    pages = [PageText(1, "a" * 120), PageText(2, "b" * 120), PageText(3, "c" * 120)]
    chunks = chunk_text("p", pages, 100, 10)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3


def test_clean_text():
    assert clean_text("hyphen-\nated  text\n\nnew") == "hyphenated text new"
    assert clean_text("") == ""


def test_metrics():
    preds = [
        {"paper_id": "a", "question": "q", "model_name": "m", "answer": 3,
         "evidence": ["c1"], "retrieved_chunks": [{"chunk_id": "c1"}]},
        {"paper_id": "a", "question": "q", "model_name": "m", "answer": 4,
         "evidence": [], "retrieved_chunks": []},
    ]
    assert metrics.citation_coverage(preds) == 0.5
    assert metrics.retrieval_hit_rate(preds) == 0.5
    assert metrics.answer_stability(preds) == 0.5
    assert metrics.citation_coverage([]) == 0.0


def test_config_model_helpers():
    assert config.model_provider("gpt-5.1") == "openai"
    assert config.model_provider("mistral-small-latest") == "mistral"
    assert config.model_output_dir("gemini-2.5-flash") == "gemini"
    with pytest.raises(ValueError):
        config.model_provider("nope")
    assert set(config.all_models()) >= {"gpt-5.1", "gemini-2.5-flash"}


def test_mistral_provider_module_exists():
    """Regression: main._build_llm imported llm.mistral_llm which did not exist."""
    from llm.mistral_llm import MistralLLM, MistralSettings
    llm = MistralLLM(MistralSettings(api_key="x", model_name="m", temperature=0.1))
    assert "mistral.ai" in str(llm._client.base_url)
