"""Embedding utilities and FAISS index persistence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

import faiss
import numpy as np

from ingest.chunker import TextChunk
from ingest.embedding_service import EmbeddingService


@dataclass(frozen=True)
class EmbeddedIndex:
    """Represents an embedded FAISS index and chunk metadata."""

    index: faiss.Index
    chunks: List[TextChunk]


def build_faiss_index(chunks: List[TextChunk], embedding_service: EmbeddingService) -> EmbeddedIndex:
    """Embed chunks and build a FAISS index."""
    texts = [c.text for c in chunks]
    embeddings = embedding_service.embed_texts(texts)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return EmbeddedIndex(index=index, chunks=chunks)


def save_index(embedded: EmbeddedIndex, index_path: Path, chunks_path: Path) -> None:
    """Persist FAISS index and chunks to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(embedded.index, str(index_path))
    chunks_payload = [c.__dict__ for c in embedded.chunks]
    chunks_path.write_text(json.dumps(chunks_payload, ensure_ascii=False), encoding="utf-8")


def load_index(index_path: Path, chunks_path: Path) -> EmbeddedIndex:
    """Load FAISS index and chunks from disk."""
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Index or chunks file not found.")
    index = faiss.read_index(str(index_path))
    raw = chunks_path.read_text(encoding="utf-8")
    chunks_list = json.loads(raw)
    chunks = [TextChunk(**item) for item in chunks_list]
    return EmbeddedIndex(index=index, chunks=chunks)


if __name__ == "__main__":
    from ingest.embedding_service import EmbeddingConfig
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import PIPELINE_CONFIG

    sample_chunks = [TextChunk(chunk_id="c1", text="hello world", page_start=1, page_end=1, char_start=0, char_end=11, paper_id="sample")]
    service = EmbeddingService(EmbeddingConfig(model_name=PIPELINE_CONFIG["embedding_model_name"], cache_path=Path("./data/embeddings_cache.sqlite")))
    embedded = build_faiss_index(sample_chunks, embedding_service=service)
    print(embedded.index.ntotal)
