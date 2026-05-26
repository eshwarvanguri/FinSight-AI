"""Document ingestion pipeline for the RAG system.

Reads PDFs / HTML / text files, extracts text, attaches metadata, chunks, and
embeds them into a vector index.

Usage:
    python -m src.llm_rag.ingestion \\
        --input-dir data/raw/filings/ \\
        --index-path artifacts/embeddings/finsight_index \\
        --chunk-size 512 --chunk-overlap 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pdfplumber
from tqdm import tqdm

from src.llm_rag.chunking import RecursiveChunker, SemanticChunker
from src.llm_rag.embeddings import EmbeddingModel
from src.utils.logger import log


def extract_pdf_text(path: Path) -> list[dict]:
    """Extract text per page, preserving page numbers."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(
                        {
                            "text": text,
                            "metadata": {
                                "source": str(path.name),
                                "page": i + 1,
                                "total_pages": len(pdf.pages),
                            },
                        }
                    )
    except Exception as e:
        log.error(f"Failed to read {path}: {e}")
    return pages


def extract_text_file(path: Path) -> list[dict]:
    """Read a plain-text or markdown file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"text": text, "metadata": {"source": str(path.name)}}]


def iter_documents(input_dir: Path) -> Iterable[dict]:
    """Yield documents from a directory of mixed file types."""
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        log.info(f"Reading {path}")
        if path.suffix.lower() == ".pdf":
            yield from extract_pdf_text(path)
        elif path.suffix.lower() in {".txt", ".md", ".html"}:
            yield from extract_text_file(path)
        else:
            log.debug(f"Skipping unsupported file: {path}")


def ingest(args: argparse.Namespace) -> None:
    """Run the full ingestion pipeline."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # 1) Load embedding model
    embedder = EmbeddingModel(model_name=args.embedding_model)

    # 2) Choose chunker
    if args.use_semantic_chunking:
        chunker = SemanticChunker(embedder=embedder, threshold=0.75)
    else:
        chunker = RecursiveChunker(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

    # 3) Read all documents
    log.info(f"Reading documents from {input_dir}")
    docs = list(iter_documents(input_dir))
    log.info(f"Loaded {len(docs)} document segments")

    # 4) Chunk
    log.info("Chunking documents...")
    chunks: list[dict] = []
    for doc in tqdm(docs, desc="chunking"):
        for piece in chunker.chunk(doc["text"]):
            chunks.append({"text": piece, "metadata": doc["metadata"]})
    log.info(f"Created {len(chunks)} chunks")

    # 5) Embed
    log.info("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(texts, batch_size=args.batch_size)

    # 6) Build vector store
    if args.vector_store == "faiss":
        from src.llm_rag.retriever import FaissStore

        store = FaissStore(dim=embedder.dimension)
        store.add(embeddings, chunks)
        store.save(index_path)
    elif args.vector_store == "chroma":
        from src.llm_rag.retriever import ChromaStore

        store = ChromaStore(persist_dir=str(index_path))
        store.add(embeddings, chunks)
    else:
        raise ValueError(f"Unknown vector store: {args.vector_store}")

    log.info(f"Index built at {index_path}")

    # Save metadata
    with open(index_path / "ingestion_meta.json", "w") as f:
        json.dump(
            {
                "n_chunks": len(chunks),
                "embedding_model": args.embedding_model,
                "chunk_size": args.chunk_size,
                "vector_store": args.vector_store,
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest documents into RAG index")
    p.add_argument("--input-dir", required=True, type=str)
    p.add_argument("--index-path", required=True, type=str)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=64)
    p.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5")
    p.add_argument("--vector-store", type=str, default="faiss", choices=["faiss", "chroma"])
    p.add_argument("--use-semantic-chunking", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    ingest(parse_args())
