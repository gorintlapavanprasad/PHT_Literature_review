"""Retrieval utilities for RAG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import re

import numpy as np

from ingest.chunker import TextChunk
from ingest.embedder import EmbeddedIndex
from ingest.embedding_service import EmbeddingService


@dataclass(frozen=True)
class RetrievalResult:
    """Represents retrieved chunks for a query."""

    query: str
    chunks: List[TextChunk]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class HybridRetriever:
    """Hybrid lexical + vector retriever."""

    def __init__(
        self,
        embedded: EmbeddedIndex,
        embedding_service: EmbeddingService,
        lexical_top_k: int,
    ) -> None:
        self._embedded = embedded
        self._embedding_service = embedding_service
        self._lexical_top_k = lexical_top_k
        self._chunk_tokens = [set(_tokenize(c.text)) for c in embedded.chunks]

    def _lexical_candidates(self, query: str) -> List[int]:
        query_tokens = set(_tokenize(query))
        scores: List[tuple[int, int]] = []
        for idx, tokens in enumerate(self._chunk_tokens):
            score = len(tokens.intersection(query_tokens))
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: (-item[1], self._embedded.chunks[item[0]].chunk_id))
        return [idx for idx, _ in scores[: self._lexical_top_k]]

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        """Retrieve top-k chunks using lexical filtering and vector reranking."""
        candidate_indices = self._lexical_candidates(query)

        if not candidate_indices:
            _, indices = self._embedded.index.search(query_embedding.reshape(1, -1), top_k)
            chunks = [self._embedded.chunks[i] for i in indices[0] if i >= 0]
            return RetrievalResult(query=query, chunks=chunks)

        distances: List[tuple[int, float]] = []
        for idx in candidate_indices:
            vector = self._embedded.index.reconstruct(idx)
            distance = float(np.linalg.norm(vector - query_embedding))
            distances.append((idx, distance))
        distances.sort(key=lambda item: (item[1], self._embedded.chunks[item[0]].chunk_id))
        top_indices = [idx for idx, _ in distances[:top_k]]
        chunks = [self._embedded.chunks[i] for i in top_indices]
        return RetrievalResult(query=query, chunks=chunks)


def embed_queries(queries: Iterable[str], embedding_service: EmbeddingService) -> np.ndarray:
    """Embed queries with centralized service."""
    return embedding_service.embed_queries(list(queries))


if __name__ == "__main__":
    print("Retriever module ready.")
