"""The vectorised retriever must return exactly what the original one-at-a-time version returned, only faster."""
import time

import faiss
import numpy as np
import pytest

from ingest.chunker import TextChunk
from ingest.embedder import EmbeddedIndex
from retrieval.retriever import HybridRetriever, _tokenize

WORDS = "game play turn skill chance solo social team score rules blind child motion tablet audio fun utility".split()


def make(n_chunks, dim=48, seed=0):
    rng = np.random.default_rng(seed)
    chunks = [TextChunk("p", f"chunk_{i}", " ".join(rng.choice(WORDS, size=30)), 1, 1, 0, 10) for i in range(n_chunks)]
    vecs = rng.normal(size=(n_chunks, dim)).astype(np.float32)
    idx = faiss.IndexFlatL2(dim)
    idx.add(vecs)
    return EmbeddedIndex(index=idx, chunks=chunks), vecs, rng


def reference(emb, query, q_emb, top_k, lexical_top_k):
    """The original algorithm, kept here as the spec."""
    toks = set(_tokenize(query))
    scored = [(i, len(set(_tokenize(c.text)) & toks)) for i, c in enumerate(emb.chunks)]
    scored = [(i, s) for i, s in scored if s > 0]
    scored.sort(key=lambda x: (-x[1], emb.chunks[x[0]].chunk_id))
    cand = [i for i, _ in scored[:lexical_top_k]]
    if not cand:
        _, ix = emb.index.search(q_emb.reshape(1, -1), top_k)
        return [emb.chunks[i].chunk_id for i in ix[0] if i >= 0]
    d = sorted(((i, float(np.linalg.norm(emb.index.reconstruct(i) - q_emb))) for i in cand), key=lambda x: (x[1], emb.chunks[x[0]].chunk_id))
    top = [i for i, _ in d[:top_k]]
    if len(top) < top_k:                                   # backfill rule added earlier
        _, extra = emb.index.search(q_emb.reshape(1, -1), top_k)
        for i in extra[0]:
            if i >= 0 and int(i) not in top:
                top.append(int(i))
            if len(top) >= top_k:
                break
    return [emb.chunks[i].chunk_id for i in top]


@pytest.mark.parametrize("n_chunks,top_k,lex_k,seed", [(40, 5, 50, 1), (120, 15, 50, 2), (120, 15, 10, 3), (60, 60, 50, 4), (25, 100, 50, 5)])
def test_same_chunks_in_same_order_as_the_original(n_chunks, top_k, lex_k, seed):
    emb, _, rng = make(n_chunks, seed=seed)
    retr = HybridRetriever(emb, None, lexical_top_k=lex_k)
    for _ in range(10):
        query = " ".join(rng.choice(WORDS, size=6))
        q_emb = rng.normal(size=48).astype(np.float32)
        got = [c.chunk_id for c in retr.retrieve(query, q_emb, top_k).chunks]
        assert got == reference(emb, query, q_emb, top_k, lex_k)


def test_query_with_no_lexical_match_falls_back_to_vector_search():
    emb, _, rng = make(30)
    retr = HybridRetriever(emb, None, lexical_top_k=50)
    q_emb = rng.normal(size=48).astype(np.float32)
    got = [c.chunk_id for c in retr.retrieve("zzzz qqqq", q_emb, 5).chunks]
    assert got == reference(emb, "zzzz qqqq", q_emb, 5, 50) and len(got) == 5


def test_empty_index_does_not_crash():
    idx = faiss.IndexFlatL2(8)
    retr = HybridRetriever(EmbeddedIndex(index=idx, chunks=[]), None, lexical_top_k=50)
    assert retr.retrieve("game", np.zeros(8, dtype=np.float32), 5).chunks == []


def test_it_is_not_slower_than_the_original_on_a_realistic_paper():
    emb, _, rng = make(300, dim=256, seed=9)
    retr = HybridRetriever(emb, None, lexical_top_k=50)
    queries = [(" ".join(rng.choice(WORDS, size=8)), rng.normal(size=256).astype(np.float32)) for _ in range(60)]
    t0 = time.perf_counter()
    for q, e in queries:
        retr.retrieve(q, e, 15)
    new = time.perf_counter() - t0
    t0 = time.perf_counter()
    for q, e in queries:
        reference(emb, q, e, 15, 50)
    old = time.perf_counter() - t0
    assert new < old * 1.1, f"vectorised {new:.3f}s vs original {old:.3f}s"
