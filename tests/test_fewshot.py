"""Issue #25: few-shot examples from consensus-coded papers, with leave-one-paper-out."""
import json

import pandas as pd
import pytest

from conftest import ScriptedLLM, llm_json
from ingest.chunker import TextChunk
from llm.fewshot import Example, FewShotBank
from llm.prompts import build_prompt
from pipeline.annotate import annotate_question
from questions import PHT_FRAMEWORK_QUESTIONS as QS
from test_pipeline_variations import _embedded


def bank(n_papers=30):
    ex = [Example(f"p{i}", q, (i % 5) + 1, f"excerpt of paper {i} about question {q}") for i in range(n_papers) for q in range(8)]
    return FewShotBank(ex)


def test_never_uses_the_paper_being_coded():
    b = bank()
    for pid in ("p0", "p7", "p29"):
        for q in range(8):
            assert all(e.paper_id != pid for e in b.select(pid, q, k=5))
    block = b.block_for("p3", 2, k=29)
    assert "excerpt of paper 3 about question 2" not in block


def test_covers_different_labels_and_is_deterministic_and_k_bounded():
    b = bank()
    sel = b.select("zzz", 1, k=5)
    assert len(sel) == 5 and {e.label for e in sel} == {1, 2, 3, 4, 5}
    assert [e.paper_id for e in sel] == [e.paper_id for e in bank().select("zzz", 1, k=5)]
    assert len(b.select("zzz", 1, k=2)) == 2 and b.select("zzz", 1, k=0) == []


def test_only_same_question_examples_and_empty_when_none_available():
    b = FewShotBank([Example("a", 0, 3, "x")])
    assert b.select("b", 5) == [] and b.block_for("b", 5) == ""
    assert b.select("a", 0) == []                         # the only example is the paper itself


def test_block_text_marks_examples_as_team_coded_and_includes_labels():
    block = bank().block_for("zzz", 0, k=3)
    assert block.startswith("WORKED EXAMPLES") and "NOT one of them" in block and block.count("Team answer:") == 3


def test_build_prompt_replaces_the_fixed_example_only_when_given(make_pdf, cfg, fake_embedder):
    emb, _ = _embedded(make_pdf, cfg, fake_embedder, "short_1page")
    default = build_prompt(QS[0], emb.chunks, 0)
    custom = build_prompt(QS[0], emb.chunks, 0, extra_examples="WORKED EXAMPLES MARKER")
    assert "WORKED EXAMPLES MARKER" in custom and "WORKED EXAMPLES MARKER" not in default
    assert custom != default


def test_annotation_uses_provider_and_passes_the_paper_id(make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "short_1page")
    seen = []

    class Spy(ScriptedLLM):
        def generate(self, prompt):
            seen.append(prompt)
            return super().generate(prompt)

    calls = []

    def provider(pid, qi):
        calls.append((pid, qi))
        return "WORKED EXAMPLES FOR THIS QUESTION"

    rec = annotate_question("paperX", emb, retr, QS[2], fake_embedder.embed_query(QS[2]), Spy([llm_json(3)]), "m", "v", 15, 2,
                            fewshot=provider)
    assert calls == [("paperX", 2)] and "WORKED EXAMPLES FOR THIS QUESTION" in seen[0] and rec.answer == 3


def test_annotation_without_provider_is_unchanged(make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "short_1page")
    seen = []

    class Spy(ScriptedLLM):
        def generate(self, prompt):
            seen.append(prompt)
            return super().generate(prompt)

    annotate_question("p", emb, retr, QS[0], fake_embedder.embed_query(QS[0]), Spy([llm_json(3)]), "m", "v", 15, 0)
    assert "WORKED EXAMPLES" not in seen[0]


def test_build_from_human_labels_and_saved_indexes(tmp_path):
    idx = tmp_path / "idx"
    idx.mkdir()
    for pid in ("a", "b"):
        chunks = [TextChunk(pid, f"chunk_{i}", f"{pid} text {i} " * 30, 1, 1, 0, 10).__dict__ for i in range(3)]
        (idx / f"{pid}.chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    res = tmp_path / "res" / "openai"
    res.mkdir(parents=True)
    (res / "a.jsonl").write_text(json.dumps({"question": QS[0], "evidence": ["chunk_2"]}) + "\n", encoding="utf-8")
    human = pd.DataFrame([{"paper_id": "a", "q": 0, "human": 2}, {"paper_id": "b", "q": 0, "human": 4},
                          {"paper_id": "missing", "q": 0, "human": 1}])
    b = FewShotBank.build(human, idx, tmp_path / "res")
    assert {e.paper_id for e in b.examples} == {"a", "b"}                      # no index -> skipped
    a = next(e for e in b.examples if e.paper_id == "a")
    assert a.excerpt.startswith("a text 2") and len(a.excerpt) <= 700          # cited chunk, truncated
    b.save(tmp_path / "bank.json")
    assert FewShotBank.load(tmp_path / "bank.json").examples == b.examples


def test_cli_flag_exists():
    import main as pipeline_main
    import sys
    old = sys.argv
    sys.argv = ["main.py", "--fewshot-human", "x.csv", "--fewshot-k", "2"]
    try:
        a = pipeline_main._parse_args()
    finally:
        sys.argv = old
    assert str(a.fewshot_human) == "x.csv" and a.fewshot_k == 2
