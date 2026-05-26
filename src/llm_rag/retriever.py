"""Vector store wrappers + hybrid retriever.

Implements:
  - FaissStore: dense vector store
  - ChromaStore: alternative with persistence built-in
  - HybridRetriever: BM25 + dense + Reciprocal Rank Fusion
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import log


class FaissStore:
    """Thin wrapper around a FAISS IndexFlatIP for cosine similarity (assuming
    normalized embeddings)."""

    def __init__(self, dim: int):
        import faiss  # local import

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        log.info(f"FAISS: added {len(chunks)} vectors. Total: {self.index.ntotal}")

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, idxs = self.index.search(query_vec, top_k)
        out: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            out.append(item)
        return out

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str | Path) -> "FaissStore":
        import faiss

        path = Path(path)
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        store = cls.__new__(cls)
        store.dim = index.d
        store.index = index
        store.metadata = metadata
        return store


class ChromaStore:
    """ChromaDB-backed store (persistent by default)."""

    def __init__(self, persist_dir: str, collection_name: str = "finsight"):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        self.collection.add(
            ids=[f"doc_{i}" for i in range(len(chunks))],
            embeddings=embeddings.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        result = self.collection.query(
            query_embeddings=[query_vec.tolist()], n_results=top_k
        )
        out = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            out.append({"text": doc, "metadata": meta, "score": 1 - dist})
        return out


# ---------------------------- BM25 ---------------------------- #

class BM25Index:
    """Simple BM25 over the same chunk corpus."""

    def __init__(self):
        from rank_bm25 import BM25Okapi

        self._BM25 = BM25Okapi
        self.bm25 = None
        self.docs: list[dict] = []

    def fit(self, chunks: list[dict]) -> None:
        self.docs = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = self._BM25(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[-top_k:][::-1]
        out = []
        for i in top_idx:
            item = dict(self.docs[i])
            item["score"] = float(scores[i])
            out.append(item)
        return out


# ---------------------------- Hybrid ---------------------------- #

class HybridRetriever:
    """Combine BM25 (lexical) and dense retrieval using Reciprocal Rank
    Fusion."""

    def __init__(
        self,
        dense_store: FaissStore,
        bm25: BM25Index | None,
        embedder,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
    ):
        self.dense = dense_store
        self.bm25 = bm25
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20) -> list[dict]:
        q_vec = self.embedder.embed(query)
        dense_hits = self.dense.search(q_vec, top_k=fetch_k)

        if self.bm25 is None:
            return dense_hits[:top_k]

        bm25_hits = self.bm25.search(query, top_k=fetch_k)
        return self._rrf(dense_hits, bm25_hits, top_k=top_k)

    def _rrf(self, list_a: list[dict], list_b: list[dict], top_k: int) -> list[dict]:
        """Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        records: dict[str, dict] = {}

        def key(item: dict) -> str:
            return f"{item['metadata'].get('source')}::{item['text'][:80]}"

        for rank, item in enumerate(list_a):
            k = key(item)
            scores[k] = scores.get(k, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
            records[k] = item
        for rank, item in enumerate(list_b):
            k = key(item)
            scores[k] = scores.get(k, 0.0) + self.bm25_weight / (self.rrf_k + rank + 1)
            records.setdefault(k, item)

        sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
        out = []
        for k in sorted_keys:
            rec = dict(records[k])
            rec["score"] = scores[k]
            out.append(rec)
        return out
