"""End-to-end RAG query interface.

Usage:
    python -m src.llm_rag.generator \\
        --query "What were the main risk factors in the 10-K?" \\
        --index-path artifacts/embeddings/finsight_index \\
        --top-k 5 \\
        --use-reranker
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.llm_rag.embeddings import EmbeddingModel
from src.llm_rag.reranker import CrossEncoderReranker
from src.llm_rag.retriever import BM25Index, FaissStore, HybridRetriever
from src.utils.config import settings
from src.utils.logger import log


SYSTEM_PROMPT = """You are FinSight, a financial document analyst.
You answer questions using ONLY the provided context from SEC filings and
financial reports.

Rules:
1. Ground every claim in the provided context.
2. Cite sources inline as [source:page].
3. If the context doesn't contain the answer, say so — do not fabricate.
4. Be precise with numbers and dates from the filings.
5. Quote figures verbatim from filings when relevant.

Context:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into LLM-ready context."""
    formatted = []
    for i, ch in enumerate(chunks, 1):
        meta = ch.get("metadata", {})
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        formatted.append(f"[{i}] (source: {src}, page: {page})\n{ch['text']}\n")
    return "\n---\n".join(formatted)


def call_llm(
    prompt: str,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """Call the LLM provider. Falls back to a stub if no API key set."""
    model = model or settings.LLM_MODEL or "grok-3-mini"
    provider = provider or settings.LLM_PROVIDER or "xai"

    if provider == "xai":
        if not settings.XAI_API_KEY:
            log.warning("No XAI_API_KEY set — returning context-only stub answer.")
            return "[LLM disabled: set XAI_API_KEY to enable generation]\n\n" + prompt[:500]
        from openai import OpenAI

        client = OpenAI(api_key=settings.XAI_API_KEY, base_url=settings.XAI_BASE_URL)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            log.warning("No OPENAI_API_KEY set — returning context-only stub answer.")
            return "[LLM disabled: set OPENAI_API_KEY to enable generation]\n\n" + prompt[:500]
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    if provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            log.warning("No ANTHROPIC_API_KEY set.")
            return "[LLM disabled]"
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    raise ValueError(f"Unknown provider: {provider}")


class RAGPipeline:
    """End-to-end RAG: retrieve → rerank → generate."""

    def __init__(
        self,
        index_path: str | Path,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        use_reranker: bool = True,
        use_hybrid: bool = True,
    ):
        self.index_path = Path(index_path)
        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.store = FaissStore.load(self.index_path)

        self.bm25: BM25Index | None = None
        if use_hybrid:
            self.bm25 = BM25Index()
            self.bm25.fit(self.store.metadata)

        self.retriever = HybridRetriever(
            dense_store=self.store,
            bm25=self.bm25,
            embedder=self.embedder,
        )
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def query(
        self,
        question: str,
        top_k: int = 5,
        llm_model: str | None = None,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        """Run the full RAG pipeline."""
        log.info(f"RAG query: {question}")

        # 1) Retrieve
        candidates = self.retriever.retrieve(question, top_k=top_k, fetch_k=20)
        log.info(f"Retrieved {len(candidates)} candidates")

        # 2) Rerank
        if self.reranker:
            candidates = self.reranker.rerank(question, candidates, top_n=top_k)

        # 3) Generate
        context = format_context(candidates)
        prompt = SYSTEM_PROMPT.format(context=context, question=question)
        answer = call_llm(prompt, model=llm_model, provider=llm_provider)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": c["metadata"].get("source"),
                    "page": c["metadata"].get("page"),
                    "score": c.get("rerank_score", c.get("score")),
                    "snippet": c["text"][:200] + "...",
                }
                for c in candidates
            ],
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Query the RAG index")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--index-path", type=str, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--use-reranker", action="store_true")
    p.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    p.add_argument("--llm-provider", type=str, default="openai",
                   choices=["openai", "anthropic"])
    p.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5")
    args = p.parse_args()

    pipeline = RAGPipeline(
        index_path=args.index_path,
        embedding_model=args.embedding_model,
        use_reranker=args.use_reranker,
    )
    result = pipeline.query(args.query, top_k=args.top_k,
                            llm_model=args.llm_model, llm_provider=args.llm_provider)

    print("\n" + "=" * 80)
    print("ANSWER:")
    print(result["answer"])
    print("\n" + "=" * 80)
    print("SOURCES:")
    for s in result["sources"]:
        print(f"  • {s['source']} p.{s['page']} (score={s['score']:.3f})")
        print(f"    {s['snippet']}")


if __name__ == "__main__":
    main()
