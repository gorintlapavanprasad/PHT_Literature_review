"""API tests for pht_ui backend: upload -> stream (fake LLMs) -> cached results -> re-stream."""
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbeddingService, ScriptedLLM, llm_json
from questions import PHT_FRAMEWORK_QUESTIONS as QS


@pytest.fixture
def client(cfg, monkeypatch):
    from backend import server
    monkeypatch.setattr(server, "_config", cfg)
    monkeypatch.setattr(server, "_embedding_service", FakeEmbeddingService())
    llms = {}

    def fake_get_llm(name):
        # each model gets its own deterministic answer so we can check routing
        base = {"gpt-5.1": 1, "gemini-2.5-flash": 2, "llama-3.3-70b-versatile": 3, "mistral-small-latest": 4}[name]
        llms.setdefault(name, ScriptedLLM([llm_json(answer=base)]))
        return llms[name]

    monkeypatch.setattr(server, "_get_llm", fake_get_llm)
    c = TestClient(server.app)
    c.llms = llms
    return c


def sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        ev, data = block.split("\n", 1)
        events.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_meta_and_empty_state(client):
    meta = client.get("/api/meta").json()
    assert [m["alias"] for m in meta["models"]] == ["openai", "gemini", "groq", "mistral"]
    assert len(meta["dimensions"]) == len(QS)
    assert client.get("/api/papers").json() == {"papers": []}
    assert client.get("/api/results/none").json()["results"] == {}


@pytest.mark.parametrize("name,ok", [("a.pdf", True), ("B C.PDF", True), ("notes.txt", False), ("x.pdf.exe", False)])
def test_upload_validation(client, make_pdf, name, ok):
    pdf = make_pdf("src", ["hello world game"])
    r = client.post("/api/upload", files={"file": (name, pdf.read_bytes(), "application/pdf")})
    assert (r.status_code == 200) is ok


def test_stream_new_paper_then_cached(client, make_pdf):
    pdf = make_pdf("newpaper", ["cooperative game for blind users " * 80] * 3)
    assert client.post("/api/upload", files={"file": ("newpaper.pdf", pdf.read_bytes())}).status_code == 200

    events = sse(client.get("/api/stream/newpaper").text)
    names = [e for e, _ in events]
    assert names[0] == "ingest_started" and "ingest_done" in names and names[-1] == "done"
    results = [p for e, p in events if e == "result"]
    assert len(results) == 4 * len(QS)
    by_model = {}
    for p in results:
        assert p["cached"] is False
        by_model.setdefault(p["model"], set()).add(p["record"]["answer"])
    assert by_model == {"openai": {1}, "gemini": {2}, "groq": {3}, "mistral": {4}}
    assert not [e for e in names if e.endswith("error")]

    # results are persisted and served by /api/results
    body = client.get("/api/results/newpaper").json()
    assert body["pdf_exists"] and set(body["results"]) == {"openai", "gemini", "groq", "mistral"}
    assert all(len(v) == len(QS) for v in body["results"].values())
    assert client.get("/api/papers").json()["papers"] == ["newpaper"]

    # second stream = all cached, zero LLM calls
    calls_before = sum(l.calls for l in client.llms.values())
    events2 = sse(client.get("/api/stream/newpaper").text)
    res2 = [p for e, p in events2 if e == "result"]
    assert len(res2) == 4 * len(QS) and all(p["cached"] for p in res2)
    assert sum(l.calls for l in client.llms.values()) == calls_before


def test_stream_resumes_partial_results(client, make_pdf, cfg):
    pdf = make_pdf("partial", ["game rules turn based play " * 100] * 2)
    client.post("/api/upload", files={"file": ("partial.pdf", pdf.read_bytes())})
    client.get("/api/stream/partial")                       # populate everything
    path = cfg.results_dir / "gemini" / "partial.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")  # keep 3 of 8
    events = sse(client.get("/api/stream/partial").text)
    gem = [p for e, p in events if e == "result" and p["model"] == "gemini"]
    assert sum(p["cached"] for p in gem) == 3 and sum(not p["cached"] for p in gem) == len(QS) - 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(QS)


def test_stream_missing_pdf_404(client):
    assert client.get("/api/stream/ghost").status_code == 404


def test_one_provider_failing_does_not_break_others(client, make_pdf, monkeypatch):
    from backend import server
    pdf = make_pdf("failmodel", ["game " * 400])
    client.post("/api/upload", files={"file": ("failmodel.pdf", pdf.read_bytes())})
    orig = server._get_llm

    def flaky(name):
        if name == "gpt-5.1":
            raise EnvironmentError("OPENAI_API_KEY is required")
        return orig(name)

    monkeypatch.setattr(server, "_get_llm", flaky)
    events = sse(client.get("/api/stream/failmodel").text)
    assert any(e == "model_error" and p["model"] == "openai" for e, p in events)
    ok_models = {p["model"] for e, p in events if e == "result"}
    assert ok_models == {"gemini", "groq", "mistral"}
    assert events[-1][0] == "done"


def test_pipeline_root_autodetected():
    from backend import server
    assert (server.PIPELINE_ROOT / "main.py").exists()


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_concurrent_persist_does_not_lose_records(cfg, tmp_path):
    """Regression: two writers for the same file used to overwrite each other (last writer wins)."""
    import threading
    from backend import server
    from pipeline.annotate import AnnotationRecord

    def rec(i):
        return AnnotationRecord("p", QS[i], "m", "v", 3, 0.5, [], [], [], "r", "retrieval", 0.0, "t")

    out = tmp_path / "res" / "openai" / "p.jsonl"
    ts = [threading.Thread(target=server._persist_records, args=(out, [rec(i)])) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(QS.index(r["question"]) for r in rows) == list(range(8))
