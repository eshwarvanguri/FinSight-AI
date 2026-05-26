#!/usr/bin/env bash
# Build the RAG index from financial filings
set -euo pipefail

echo "→ Downloading sample 10-K texts..."
python scripts/download_data.py --dataset sample_10k

echo "→ Building RAG index..."
python -m src.llm_rag.ingestion \
    --input-dir data/raw/filings/ \
    --index-path artifacts/embeddings/finsight_index \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-model BAAI/bge-small-en-v1.5 \
    --vector-store faiss

echo "✓ Done. Index: artifacts/embeddings/finsight_index/"
