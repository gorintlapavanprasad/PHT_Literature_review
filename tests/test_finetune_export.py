import json

import pandas as pd

from ingest.chunker import TextChunk
from questions import PHT_FRAMEWORK_QUESTIONS as QS
from training import export_finetune as E


def _index(tmp, pid, n=20):
    d = tmp / "idx"
    d.mkdir(exist_ok=True)
    chunks = [TextChunk(pid, f"chunk_{i}", f"text {i} cooperative game", 1, 1, i * 10, i * 10 + 9).__dict__
              for i in range(n)]
    (d / f"{pid}.chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return d


def test_split_is_deterministic_and_roughly_balanced():
    assert E.split_of("abc") == E.split_of("abc")
    vals = [E.split_of(f"paper{i}") for i in range(1000)]
    frac = vals.count("val") / 1000
    assert 0.12 < frac < 0.28


def test_export_roundtrip_and_no_paper_leakage(tmp_path):
    papers = [f"p{i}" for i in range(30)]
    for p in papers:
        idx = _index(tmp_path, p)
    human = pd.DataFrame([{"paper_id": p, "q": q, "human": (q % 5) + 1} for p in papers for q in range(8)])
    human = pd.concat([human, pd.DataFrame([{"paper_id": "no_index", "q": 0, "human": 3}])])
    counts = E.export(human, idx, tmp_path / "out")
    assert counts["train"] + counts["val"] == 240 and counts["skipped_no_chunks"] == 1
    tr = [json.loads(l) for l in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    va = [json.loads(l) for l in (tmp_path / "out" / "val.jsonl").read_text().splitlines()]
    assert {e["meta"]["paper_id"] for e in tr}.isdisjoint({e["meta"]["paper_id"] for e in va})
    ex = tr[0]
    assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
    target = json.loads(ex["messages"][1]["content"])
    assert target["answer"] == (ex["meta"]["q"] % 5) + 1 and 1 <= target["answer"] <= 5
    assert QS[ex["meta"]["q"]].strip()[:40] in ex["messages"][0]["content"]


def test_evidence_uses_reference_llm_chunks(tmp_path):
    idx = _index(tmp_path, "pa")
    res = tmp_path / "res" / "openai"
    res.mkdir(parents=True)
    (res / "pa.jsonl").write_text(json.dumps({"question": QS[0], "evidence": ["chunk_7", "chunk_bad"]}) + "\n")
    rows = [("pa", 0, 4)]
    (_, ex), = list(E.build_examples(rows, idx, tmp_path / "res"))
    target = json.loads(ex["messages"][1]["content"])
    assert target["evidence"] == ["chunk_7"]                       # unknown ids dropped
    assert "text 7" in ex["messages"][0]["content"]               # cited chunk is in the prompt
