"""Central embedding service with caching."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding service."""

    model_name: str
    cache_path: Path


class EmbeddingService:
    """Embedding service with content-based caching."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._model = SentenceTransformer(config.model_name)
        self._cache_path = config.cache_path
        self._cache: dict[str, np.ndarray] = {}
        self._model_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    hash TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, text_hash: str) -> np.ndarray | None:
        if text_hash in self._cache:
            return self._cache[text_hash]

        with self._db_lock, sqlite3.connect(self._cache_path) as conn:
            row = conn.execute(
                "SELECT dim, embedding FROM embeddings WHERE hash = ?", (text_hash,)
            ).fetchone()
            if row is None:
                return None
            dim, blob = row
            embedding = np.frombuffer(blob, dtype=np.float32)
            embedding = embedding.reshape(dim)
            self._cache[text_hash] = embedding
            return embedding

    def _set_cached(self, text_hash: str, embedding: np.ndarray) -> None:
        embedding = embedding.astype(np.float32)
        self._cache[text_hash] = embedding
        with self._db_lock, sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (hash, dim, embedding) VALUES (?, ?, ?)",
                (text_hash, embedding.shape[0], embedding.tobytes()),
            )
            conn.commit()

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Embed texts with caching. Returns a 2D numpy array."""
        text_list = list(texts)
        hashes = [self._hash_text(text) for text in text_list]

        embeddings: List[np.ndarray | None] = [None] * len(text_list)
        to_compute: List[str] = []
        to_compute_indices: List[int] = []

        for i, text_hash in enumerate(hashes):
            cached = self._get_cached(text_hash)
            if cached is None:
                to_compute.append(text_list[i])
                to_compute_indices.append(i)
            else:
                embeddings[i] = cached

        if to_compute:
            logging.info("Embedding %s uncached texts", len(to_compute))
            with self._model_lock:
                computed = self._model.encode(to_compute, convert_to_numpy=True)
            for idx, embedding in zip(to_compute_indices, computed):
                text_hash = hashes[idx]
                self._set_cached(text_hash, embedding)
                embeddings[idx] = embedding

        if any(e is None for e in embeddings):
            raise RuntimeError("Embedding cache mismatch in embed_texts")
        result = np.stack([e for e in embeddings], axis=0).astype(np.float32)
        return result

    def embed_queries(self, queries: Iterable[str]) -> np.ndarray:
        """Embed queries with caching."""
        return self.embed_texts(list(queries))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with caching."""
        return self.embed_texts([query])[0]
