"""End-to-end variations: new PDFs are imported, ingested, annotated and re-read from disk.

Uses real PDF parsing, real chunking, real FAISS, real retriever, real JSONL writer;
only the embedding model and the LLM are faked (no torch / API keys required).
"""
import json

import pytest

import main as pipeline_main
from conftest import ScriptedLLM, llm_json
from ingest.embedder import load_index
from pipeline.annotate import annotate_question
from questions import PHT_FRAMEWORK_QUESTIONS as QS
from retrieval.retriever import HybridRetriever
from storage.writer import write_jsonl

LOREM = ("Participants played a cooperative tablet game with blind users. "
         "The game used turn based rules and scoring. ") * 40

# 12 paper variations: (name, pages)
PAPERS = {
    "short_1page": ["A tiny paper about a cooperative game for blind players."],
    "medium_5pages": [LOREM] * 5,
    "long_25pages": [LOREM + f" page {i}" for i in range(25)],
    "empty_pdf": [""],
    "blank_pages": ["", "", ""],
    "unicode": ["Zusammenarbeit Spiel für Sehbehinderte — café naïve 日本語 ✓ " * 30],
    "hyphenated": ["accessi-\nbility is very impor-\ntant in game design " * 50],
    "numbers_only": ["1 2 3 4 5 6 7 8 9 " * 80],
    "one_huge_page": [LOREM * 3],
    "repeated_text": ["same same same same " * 200],
    "survey_not_game": ["We surveyed 200 clinicians about their workflow and tools. " * 60],
    "mixed_empty_and_text": ["", LOREM, "", LOREM],
}


def run_paper(pdf, cfg, embedder, llm, model="gpt-5.1"):
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    pipeline_main.ingest_paper(pdf, cfg, embedder)
    q_emb = embedder.embed_queries(QS)
    recs = pipeline_main.annotate_paper(pdf, QS, q_emb, llm, embedder, cfg, model)
    return recs


@pytest.mark.parametrize("name", list(PAPERS))
def test_paper_variation_roundtrip(name, make_pdf, cfg, fake_embedder, tmp_path):
    pdf = make_pdf(name, PAPERS[name])
    chunks_path = cfg.index_dir / f"{name}.chunks.json"
    llm = ScriptedLLM([llm_json(answer=(i % 5) + 1, evidence=["chunk_0"]) for i in range(8)])
    try:
        recs = run_paper(pdf, cfg, fake_embedder, llm)
    except Exception as exc:
        # An empty/blank PDF has zero chunks; FAISS cannot embed nothing. That must be a
        # clean, explainable failure (not a crash deep in numpy) - recorded as a known gap.
        assert name in {"empty_pdf", "blank_pages"}, f"unexpected failure for {name}: {exc!r}"
        assert isinstance(exc, ValueError) and "No extractable text" in str(exc)
        return
    assert len(recs) == len(QS)
    out = cfg.results_dir / "openai" / f"{name}.jsonl"
    write_jsonl(recs, out)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert {r["question"] for r in rows} == set(QS)
    for r in rows:
        assert r["answer"] in (1, 2, 3, 4, 5, "Unknown", "Not Applicable")
        assert 0.0 <= r["confidence"] <= 1.0
    # index round-trips from disk
    emb = load_index(cfg.index_dir / f"{name}.faiss", chunks_path)
    assert emb.index.ntotal == len(emb.chunks) > 0


def _embedded(make_pdf, cfg, fake_embedder, name="medium_5pages"):
    pdf = make_pdf(name, PAPERS[name])
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    pipeline_main.ingest_paper(pdf, cfg, fake_embedder)
    emb = load_index(cfg.index_dir / f"{name}.faiss", cfg.index_dir / f"{name}.chunks.json")
    return emb, HybridRetriever(emb, fake_embedder, lexical_top_k=cfg.lexical_top_k)


def _ask(emb, retr, emb_svc, llm, q=0, top_k=15):
    return annotate_question("p", emb, retr, QS[q], emb_svc.embed_query(QS[q]), llm, "m", "v", top_k, q)


@pytest.mark.parametrize("script,answer,mode", [
    ([llm_json(3)], 3, "retrieval"),
    (['```json\n{"answer": 4, "confidence": 0.7, "evidence": ["chunk_1"], "reasoning": "r"}\n```'], 4, "retrieval"),
    (['{"answer": "2", "confidence": 0.7, "evidence": ["chunk_1"], "reasoning": "r"}'], 2, "retrieval"),
    ([llm_json("Not Applicable")], "Not Applicable", "not_applicable"),
    (["total garbage", "still garbage"], "Unknown", "unresolved"),
    ([llm_json("Unknown"), llm_json(5)], 5, "expanded_retry"),
    ([llm_json("Unknown"), llm_json("Not Applicable")], "Not Applicable", "not_applicable"),
    ([llm_json("Unknown"), llm_json("Unknown")], "Unknown", "unresolved"),
    ([llm_json(3, confidence="high")], 3, "retrieval"),
    ([llm_json(3, evidence="chunk_0")], 3, "retrieval"),
])
def test_llm_response_variations_on_long_paper(script, answer, mode, make_pdf, cfg, fake_embedder):
    # long paper => retrieval path (more than 30 chunks)
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "long_25pages")
    assert len(emb.chunks) > 30
    rec = _ask(emb, retr, fake_embedder, ScriptedLLM(script))
    assert rec.answer == answer
    assert rec.decision_mode == mode
    assert len(rec.retrieved_chunks) > 0


def test_short_paper_uses_full_context(make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "short_1page")
    assert len(emb.chunks) <= 30
    rec = _ask(emb, retr, fake_embedder, ScriptedLLM([llm_json(2, evidence=["chunk_0"])]))
    assert rec.decision_mode == "full_context" and len(rec.retrieved_chunks) == len(emb.chunks)
    assert rec.evidence_metadata and rec.evidence_metadata[0]["chunk_id"] == "chunk_0"


def test_resolved_answer_without_evidence_is_penalised(make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "long_25pages")
    rec = _ask(emb, retr, fake_embedder, ScriptedLLM([llm_json(3, 0.8, evidence=[])]))
    assert rec.confidence == pytest.approx(0.4)


@pytest.mark.parametrize("top_k", [1, 5, 15, 40, 500])
def test_retriever_returns_exactly_topk_when_available(top_k, make_pdf, cfg, fake_embedder):
    """Regression: a narrow lexical filter used to return < top_k chunks."""
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "long_25pages")
    res = retr.retrieve(QS[0], fake_embedder.embed_query(QS[0]), top_k)
    assert len(res.chunks) == min(top_k, len(emb.chunks))
    assert len({c.chunk_id for c in res.chunks}) == len(res.chunks)  # no duplicates


def test_retriever_finds_relevant_chunk(make_pdf, cfg, fake_embedder):
    pages = ["filler words about nothing " * 40] * 6 + ["dice chance Catan board randomness " * 40]
    pdf = make_pdf("needle", pages)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    pipeline_main.ingest_paper(pdf, cfg, fake_embedder)
    emb = load_index(cfg.index_dir / "needle.faiss", cfg.index_dir / "needle.chunks.json")
    retr = HybridRetriever(emb, fake_embedder, lexical_top_k=50)
    q = "dice chance Catan randomness"
    res = retr.retrieve(q, fake_embedder.embed_query(q), 3)
    assert any("dice" in c.text for c in res.chunks[:2])


def test_ingest_is_idempotent_and_force_rebuilds(make_pdf, cfg, fake_embedder):
    pdf = make_pdf("idem", PAPERS["medium_5pages"])
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    p1 = pipeline_main.ingest_paper(pdf, cfg, fake_embedder)
    m1 = p1.stat().st_mtime_ns
    pipeline_main.ingest_paper(pdf, cfg, fake_embedder)           # cached
    assert p1.stat().st_mtime_ns == m1
    pipeline_main.ingest_paper(pdf, cfg, fake_embedder, force=True)  # rebuilt
    assert p1.stat().st_mtime_ns >= m1


def test_resolve_models_aliases():
    for alias in ("openai", "gemini", "groq", "mistral", "llama"):
        assert pipeline_main._resolve_models([alias], pipeline_main.all_models())
    with pytest.raises(SystemExit):
        pipeline_main._resolve_models(["nonsense"], pipeline_main.all_models())


# ---------------------------------------------------------------- iterative expansion (paper Sec. 3.3.2)
def _ask_exp(emb, retr, svc, llm, max_expansions, top_k=5):
    return annotate_question("p", emb, retr, QS[0], svc.embed_query(QS[0]), llm, "m", "v", top_k, 0,
                             max_expansions=max_expansions)


@pytest.mark.parametrize("script,max_exp,answer,mode,calls", [
    ([llm_json("Unknown"), llm_json("Unknown"), llm_json(4)], 2, 4, "expanded_retry", 3),
    ([llm_json("Unknown"), llm_json("Unknown"), llm_json(4)], 1, "Unknown", "unresolved", 2),
    ([llm_json("Unknown"), llm_json(2)], 2, 2, "expanded_retry", 2),            # stops at first answer
    ([llm_json("Unknown"), llm_json("Unknown"), llm_json("Unknown")], 2, "Unknown", "unresolved", 3),
    ([llm_json("Unknown"), llm_json("Unknown"), llm_json("Not Applicable")], 2, "Not Applicable", "not_applicable", 3),
    ([llm_json(3)], 5, 3, "retrieval", 1),                                        # no expansion needed
    ([llm_json("Unknown")], 0, "Unknown", "unresolved", 1),                      # expansion disabled
])
def test_iterative_expansion(script, max_exp, answer, mode, calls, make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "long_25pages")
    llm = ScriptedLLM(script)
    rec = _ask_exp(emb, retr, fake_embedder, llm, max_exp)
    assert (rec.answer, rec.decision_mode, llm.calls) == (answer, mode, calls)


def test_expansion_widens_context_each_step(make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "long_25pages")
    rec = _ask_exp(emb, retr, fake_embedder, ScriptedLLM([llm_json("Unknown")] * 3), 2, top_k=5)
    assert len(rec.retrieved_chunks) == 20            # 5 -> 10 -> 20


def test_expansion_stops_when_all_chunks_used(make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "long_25pages")
    n = len(emb.chunks)
    llm = ScriptedLLM([llm_json("Unknown")])
    _ask_exp(emb, retr, fake_embedder, llm, 10, top_k=n)       # already at the max
    assert llm.calls == 1


def test_full_context_not_applicable_mode(make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "short_1page")
    rec = _ask(emb, retr, fake_embedder, ScriptedLLM([llm_json("Not Applicable")]))
    assert rec.decision_mode == "not_applicable"


@pytest.mark.parametrize("script,max_exp,passes", [
    ([llm_json(3)], 2, 1),
    ([llm_json("Unknown"), llm_json(4)], 2, 2),
    ([llm_json("Unknown"), llm_json("Unknown"), llm_json(4)], 2, 3),
    ([llm_json("Unknown")], 2, 3),
])
def test_passes_field_counts_llm_calls(script, max_exp, passes, make_pdf, cfg, fake_embedder):
    emb, retr = _embedded(make_pdf, cfg, fake_embedder, "long_25pages")
    llm = ScriptedLLM(script)
    rec = _ask_exp(emb, retr, fake_embedder, llm, max_exp)
    assert rec.passes == passes == llm.calls
