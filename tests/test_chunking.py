"""Tests for chunking strategies."""
import pytest

from src.llm_rag.chunking import RecursiveChunker


def test_recursive_chunker_short_text():
    c = RecursiveChunker(chunk_size=512, chunk_overlap=64)
    text = "Short text that fits in one chunk."
    chunks = c.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_recursive_chunker_empty():
    c = RecursiveChunker()
    assert c.chunk("") == []
    assert c.chunk("   ") == []


def test_recursive_chunker_long_text():
    c = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    text = ". ".join([f"This is sentence number {i}" for i in range(50)])
    chunks = c.chunk(text)
    assert len(chunks) > 1
    assert all(len(ch) <= 200 for ch in chunks)  # approx, with overlap allowance
