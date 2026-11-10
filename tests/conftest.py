"""Shared pytest fixtures: path setup, torch-free fake embedder, fake LLMs, synthetic PDFs."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "paper_annotation"
UI = REPO / "pht_ui"
for p in (str(PIPELINE), str(UI)):
    if p not in sys.path:
        sys.path.insert(0, p)

# sentence-transformers pulls in torch (GBs). Tests use FakeEmbeddingService instead.
if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = object
        sys.modules["sentence_transformers"] = stub

DIM = 128


class FakeEmbeddingService:
    """Deterministic hashed bag-of-words embedder: similar words => similar vectors."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in re.split(r"[^a-z0-9]+", text.lower()):
            if tok:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_texts(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_queries(self, queries):
        return self.embed_texts(list(queries))

    def embed_query(self, q):
        return self._vec(q)


class ScriptedLLM:
    """LLM stub that returns a scripted raw string per call (cycles through `script`)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        out = self.script[self.calls % len(self.script)]
        self.calls += 1
        return out if isinstance(out, str) else json.dumps(out)


def llm_json(answer=3, confidence=0.8, evidence=("chunk_0",), reasoning="because"):
    return {"answer": answer, "confidence": confidence, "evidence": list(evidence),
            "reasoning": reasoning, "latency_seconds": 0.01}


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingService()


@pytest.fixture
def make_pdf(tmp_path):
    """Factory: make_pdf(name, pages=[str,...]) -> Path of a real PDF."""
    import fitz

    def _make(name: str, pages: list[str]) -> Path:
        doc = fitz.open()
        expected_chars = 0
        for text in pages:
            # insert_textbox silently draws nothing on overflow, so paginate long text.
            segments = [text[i:i + 800] for i in range(0, len(text), 800)] or [""]
            for seg in segments:
                page = doc.new_page()
                if seg:
                    rc = page.insert_textbox(fitz.Rect(40, 40, 555, 800), seg, fontsize=9)
                    assert rc >= 0, "test PDF text overflowed its page"
                    expected_chars += len(seg.strip())
        path = tmp_path / f"{name}.pdf"
        doc.save(str(path))
        doc.close()
        if expected_chars:
            with fitz.open(str(path)) as check:
                assert sum(len(p.get_text("text").strip()) for p in check) > 0, "PDF has no extractable text"
        return path

    return _make


@pytest.fixture
def cfg(tmp_path):
    """Config whose data/pdf/results dirs live in tmp_path (never touches the repo)."""
    import dataclasses
    from config import load_config

    base = load_config()
    data = tmp_path / "data"
    return dataclasses.replace(
        base,
        data_dir=data,
        pdf_dir=tmp_path / "papers",
        index_dir=data / "indexes",
        results_dir=data / "results",
        embedding_cache_path=data / "cache.sqlite",
    )
