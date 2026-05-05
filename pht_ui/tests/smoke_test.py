"""
Headless smoke test for the PHT ScholarLens UI backend.

Verifies that the FastAPI layer correctly:
  1. Imports the existing pipeline
  2. Parses cached JSONL result files from data/results/<alias>/
  3. Maps each record to its question index via string match
  4. Returns well-formed responses from /api/meta and /api/results
  5. Computes the frontend's agreement heuristic on real data

Run this BEFORE starting the server. It proves the wiring works even when you
don't have API keys configured (no stream endpoint is touched).

Usage:
    cd pht_ui
    python tests/smoke_test.py [paper_id]
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# Make the backend importable when run from the repo root or pht_ui/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# OPTIONAL: stub sentence_transformers so this test can run on machines that
# don't have torch installed. We are testing the web layer; the embedding
# model is never actually invoked by the endpoints under test (/api/meta,
# /api/papers, /api/results). If the real package is installed, this is a
# no-op.
# ---------------------------------------------------------------------------
if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sentence_transformers")
        class _FakeST:
            def __init__(self, *a, **kw): pass
            def encode(self, *a, **kw): raise RuntimeError("stubbed — embedding not available in test")
        stub.SentenceTransformer = _FakeST
        sys.modules["sentence_transformers"] = stub

# Import the FastAPI app and its helpers. This transitively imports the
# pipeline, so if paths are broken we see it immediately.
try:
    from backend import server  # noqa: E402
except Exception as exc:
    print(f"[FAIL] could not import backend.server: {exc}")
    raise

from fastapi.testclient import TestClient


def section(title: str) -> None:
    print(f"\n\033[1m── {title} ──\033[0m")


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def warn(msg: str) -> None:
    print(f"  \033[33m!\033[0m {msg}")


def fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def main(paper_id: str = "ahmetovic21") -> int:
    failures = 0

    client = TestClient(server.app)

    # ---- 1. /api/meta returns expected shape -----------------------------
    section("meta endpoint")
    resp = client.get("/api/meta")
    if resp.status_code != 200:
        fail(f"status={resp.status_code}")
        return 1
    meta = resp.json()
    for key in ("dimensions", "models", "existing_papers"):
        if key not in meta:
            fail(f"missing key: {key}")
            failures += 1
    ok(f"dimensions: {len(meta['dimensions'])} rows")
    ok(f"models: {', '.join(m['alias'] for m in meta['models'])}")
    ok(f"existing_papers on disk: {len(meta['existing_papers'])}")

    # Sanity: UI expects 4 models in a stable order
    aliases = [m["alias"] for m in meta["models"]]
    if aliases != ["openai", "gemini", "llama", "mistral"]:
        fail(f"unexpected model order: {aliases}")
        failures += 1
    else:
        ok("model column order matches UI expectations")

    # ---- 2. /api/papers lists the cached corpus --------------------------
    section("papers endpoint")
    resp = client.get("/api/papers")
    if resp.status_code != 200:
        fail(f"status={resp.status_code}")
        return 1
    papers = resp.json()["papers"]
    ok(f"{len(papers)} papers have at least one cached result")
    if paper_id not in papers:
        warn(f"{paper_id} not in cached corpus — results endpoint will be empty")

    # ---- 3. /api/results/<pid> for a known-good paper --------------------
    section(f"results endpoint ({paper_id})")
    resp = client.get(f"/api/results/{paper_id}")
    if resp.status_code != 200:
        fail(f"status={resp.status_code}")
        return 1
    body = resp.json()
    results = body.get("results", {})
    ok(f"pdf_exists on disk: {body.get('pdf_exists')}")

    if not results:
        fail("no cached results returned — something is wrong with JSONL parsing")
        return 1

    expected_questions = len(server.PHT_FRAMEWORK_QUESTIONS)
    for alias, by_q in results.items():
        cnt = len(by_q)
        if cnt == expected_questions:
            ok(f"{alias}: {cnt}/{expected_questions} questions")
        else:
            warn(f"{alias}: {cnt}/{expected_questions} questions (partial)")

    # ---- 4. Record shape sanity on one cell ------------------------------
    section("record shape")
    first_alias = next(iter(results))
    first_q = next(iter(results[first_alias]))
    rec = results[first_alias][str(first_q)] if isinstance(first_q, str) else results[first_alias][first_q]

    # Expected fields the frontend reads
    required = ["paper_id", "question", "answer", "confidence", "evidence",
                "evidence_metadata", "retrieved_chunks", "reasoning",
                "decision_mode", "model_name"]
    missing = [f for f in required if f not in rec]
    if missing:
        fail(f"record missing fields: {missing}")
        failures += 1
    else:
        ok(f"all {len(required)} frontend-consumed fields present")

    ok(f"example cell: {first_alias} Q{first_q} → answer={rec['answer']} "
       f"confidence={rec['confidence']} mode={rec['decision_mode']}")
    ok(f"evidence: {len(rec['evidence'])} cited, "
       f"{len(rec['retrieved_chunks'])} retrieved")

    # ---- 5. Inter-model agreement mimics the frontend --------------------
    section("inter-model agreement (replicates frontend logic)")
    # For each of the 9 questions, collect numeric answers across models.
    agreement_rows = []
    # Normalize integer keys — JSON dict keys come back as strings.
    def int_keyed(d):
        return {int(k): v for k, v in d.items()}

    normalized = {alias: int_keyed(by_q) for alias, by_q in results.items()}

    for qi in range(expected_questions):
        answers = []
        for alias in ["openai", "gemini", "llama", "mistral"]:
            rec = normalized.get(alias, {}).get(qi)
            if rec and isinstance(rec.get("answer"), int):
                answers.append(rec["answer"])

        if len(answers) < 2:
            label = "—"
        else:
            counts = {}
            for a in answers:
                counts[a] = counts.get(a, 0) + 1
            max_count = max(counts.values())
            share = max_count / len(answers)
            if len(counts) == 1:
                label = f"all agree on {answers[0]}"
            elif share >= 0.75:
                label = f"{max_count}/{len(answers)} agree"
            else:
                label = f"split ({','.join(str(k) for k in sorted(counts))})"
        agreement_rows.append((qi + 1, server.PHT_DIMENSIONS[qi]["short"], answers, label))

    for qi, short, answers, label in agreement_rows:
        ans_str = ",".join(str(a) for a in answers) if answers else "—"
        print(f"  Q{qi} {short:<22} [{ans_str:<11}]  → {label}")

    # ---- summary ---------------------------------------------------------
    section("summary")
    if failures == 0:
        ok("all checks passed — backend wiring is sound")
        return 0
    else:
        fail(f"{failures} failures")
        return 1


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "ahmetovic21"
    sys.exit(main(pid))
