"""Cross-encoder reranker for precision-boosting top-k results."""
from __future__ import annotations

from typing import Iterable

from src.utils.logger import log


class CrossEncoderReranker:
    """Wrap a HF cross-encoder for query-document relevance scoring."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            log.info(f"Loading reranker: {self.model_name}")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """Re-score and re-rank candidates by cross-encoder relevance."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ordered = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ordered[:top_n]
