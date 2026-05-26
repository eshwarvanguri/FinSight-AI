"""Embedding model wrapper supporting HuggingFace + OpenAI."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from tqdm import tqdm

from src.utils.logger import log


class EmbeddingModel:
    """Lazy-loaded embedding model."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dim: int | None = None

    @property
    def model(self):
        if self._model is None:
            log.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            _ = self.model
        return self._dim  # type: ignore[return-value]

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype="float32")

    def embed_batch(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")
