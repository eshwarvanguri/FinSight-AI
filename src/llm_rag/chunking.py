"""Document chunking strategies."""
from __future__ import annotations

import re
from typing import Protocol

import numpy as np


class Chunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...


class RecursiveChunker:
    """Recursive character-based splitter (LangChain-style)."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> list[str]:
        """Recursively split text into chunks of ~chunk_size characters."""
        if not text or not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        rest = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            # hard split by char count
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            piece = (current + sep + part) if current else part
            if len(piece) <= self.chunk_size:
                current = piece
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, rest))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)

        # Add overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append(prev_tail + " " + chunks[i])
            chunks = overlapped

        return chunks


class SemanticChunker:
    """Embedding-based semantic chunker — splits at sentence-level
    similarity dips."""

    def __init__(self, embedder, threshold: float = 0.75, min_chunk: int = 100):
        self.embedder = embedder
        self.threshold = threshold
        self.min_chunk = min_chunk

    def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        embeddings = self.embedder.embed_batch(sentences)
        # cosine similarity between consecutive sentences
        sims = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        chunks = []
        current = [sentences[0]]
        for i, sim in enumerate(sims):
            if sim < self.threshold and len(" ".join(current)) > self.min_chunk:
                chunks.append(" ".join(current))
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # naive sentence splitter
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
