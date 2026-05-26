# FinSight AI — Architecture & Design Decisions

## Overview

FinSight AI is a layered platform with three core capabilities orchestrated by a fourth (the agent). The design prioritizes **modularity** (each layer is independently usable), **observability** (MLflow + structured logs), and **evaluation rigor** (both ML metrics and LLM-as-judge).

## Layer 1 — Traditional ML (Credit Risk)

**Why it exists.** Default-probability prediction is the heart of fintech lending. Random Forest / XGBoost remain the industry standard because they:
- Handle mixed feature types natively
- Are robust to outliers and missing values
- Provide tree-based feature importance (and SHAP-compatible)
- Calibrate well with isotonic regression

**Pipeline.**
1. Ingest raw loan data (LendingClub schema)
2. Clean: drop nulls, deduplicate, parse string-encoded fields
3. Engineer 8+ derived features (income-to-loan ratio, utilization buckets, etc.)
4. Stratified train/val/test split
5. ColumnTransformer (median impute + one-hot)
6. SMOTE for class imbalance
7. Stacked ensemble: XGBoost + RF + Logistic
8. Isotonic calibration on validation set
9. Evaluate: ROC-AUC, PR-AUC, KS, Brier score
10. SHAP analysis on test set

**Key trade-offs.**
- We use SMOTE only on training data (never on val/test) to avoid leakage.
- Calibration is done on a held-out val set with `cv='prefit'` — better than fitting calibration during cross-validation when the base model is already trained.
- Decision threshold is tuned on PR curve, not the default 0.5.

## Layer 2 — RAG (Document Intelligence)

**Why it exists.** Financial filings (10-K, 10-Q) contain dense, structured information that pure LLMs can't memorize and can't reliably retrieve. RAG lets us ground LLM outputs in citable, page-level evidence.

**Pipeline.**
1. Ingest PDFs (pdfplumber) with page-level metadata
2. Chunk: recursive (default) or semantic
3. Embed with BGE-small (good cost/quality trade)
4. Store in FAISS (cosine, normalized)
5. Hybrid retrieval: BM25 + dense → Reciprocal Rank Fusion
6. Cross-encoder rerank (BGE-reranker-base) for precision
7. Generate with LLM (citation-required prompt)

**Key trade-offs.**
- Hybrid > pure dense: financial documents have many exact phrases (company names, dollar figures) that BM25 captures better than embeddings.
- RRF over weighted sum: more robust to score scale differences.
- Cross-encoder rerank adds ~200ms but +15-20% precision on factual queries.

## Layer 3 — Agent (Orchestration)

**Why it exists.** Real analyst questions blend multiple capabilities: "Is X creditworthy given their 10-K?" needs both the risk model and the RAG index. A tool-using agent picks the right one(s) automatically.

**Design.**
- OpenAI function-calling (deterministic JSON tool calls > ReAct text parsing)
- 4 tools: predict_credit_risk, search_filings, calculate_financial_ratios, generate_report
- Max iterations cap prevents runaway loops
- Trace logged for every step (debugging + eval)

## Layer 4 — Serving

**FastAPI** chosen for:
- Async I/O (RAG + LLM calls are latency-bound)
- Automatic OpenAPI docs (huge for partner integrations)
- Pydantic validation (type safety + clear error messages)
- Mature ecosystem (slowapi for rate limits, Prometheus middleware, etc.)

**Lifespan loader** preloads all models once at startup — avoids reloading on every request.

## Evaluation

| Layer | Metrics |
|---|---|
| Credit risk | ROC-AUC, PR-AUC, KS, Brier, calibration plot |
| RAG | Faithfulness, answer relevance, context precision/recall (RAGAS) |
| Agent | Task success rate, avg iterations, tool-selection accuracy |

## Future Work

- LightGBM as an additional base learner
- ColBERT / late-interaction retrieval for higher precision
- Streaming responses for the agent
- A/B test framework for prompts and retrievers
- Online learning loop for risk model
